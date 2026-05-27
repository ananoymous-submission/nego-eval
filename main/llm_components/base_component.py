import os
import re
import dspy
from typing import Optional, Type
from langsmith import traceable, get_current_run_tree


def _env_prefix(component_name: str) -> str:
    """`BiddingStrategy` -> `BIDDING_STRATEGY`. Per-component env var prefix."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", component_name)
    return snake.upper()


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if val is None or val == "":
        raise ValueError(f"Required environment variable {key} is not set.")
    return val


def _parse_bool(val: str, key: str) -> bool:
    v = val.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key} must be a boolean (1/0/true/false); got {val!r}")


class Component(dspy.Module):
    """Simplified base DSPy component for LangGraph agent nodes.

    `model_name` is REQUIRED at construction time (passed through from the
    tournament's config so we can run many models in one tournament).

    The remaining settings are read from environment variables:
        <COMPONENT_NAME>_TEMPERATURE
        <COMPONENT_NAME>_MAX_TOKENS
        <COMPONENT_NAME>_REASONING      (1/0/true/false)
    All three are required. There are no fallbacks — missing env vars raise.
    """

    def __init__(
        self,
        component_name: str,
        signature: Optional[Type[dspy.Signature]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        super().__init__()
        self.component_name = component_name
        self.signature = signature

        if not model_name:
            raise ValueError(
                f"Component '{component_name}' requires model_name at construction time."
            )
        self.model_name = model_name

        prefix = _env_prefix(component_name)
        self.temperature = float(_require_env(f"{prefix}_TEMPERATURE"))
        self.max_tokens = int(_require_env(f"{prefix}_MAX_TOKENS"))
        self.reasoning = _parse_bool(_require_env(f"{prefix}_REASONING"), f"{prefix}_REASONING")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.lm = dspy.LM(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=self.max_tokens
        )

        if self.signature:
            if self.reasoning:
                self.predictor = dspy.ChainOfThought(self.signature)
            else:
                self.predictor = dspy.Predict(self.signature)

    @traceable(run_type="llm")
    def forward(self, **kwargs):
        if not self.predictor:
            raise NotImplementedError("Component must have a signature and predictor")

        result = self.predictor(**kwargs, lm=self.lm)

        rt = get_current_run_tree()
        if rt and hasattr(self.lm, 'history') and self.lm.history:
            latest_call = self.lm.history[-1]
            usage = latest_call.get('usage', {})
            cost = latest_call.get('cost', 0)

            rt.metadata['ls_model_name'] = self.model_name

            if usage:
                rt.metadata['usage_metadata'] = {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }

            if cost:
                rt.metadata['cost'] = cost

        return result
