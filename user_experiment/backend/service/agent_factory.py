"""Factory for creating agents.

The experiment compares two AI opponents the human negotiates against:
  - LLM:         `main.LLMAgent.LLMAgent` (LLM picks the bid each turn).
                 Wrapped so the natural-language message is written by the
                 SAME `DialogueGenerator` the Traditional path uses — the
                 message-writing layer is identical across conditions, so
                 the only difference observers can attribute to is which
                 process picked the bid.
  - Traditional: `main.HybridAgent.LLMAgent` (heuristic picks the bid;
                 the LLM only writes dialogue around it).

Both opponents pick a full bid that addresses every issue each turn.
"""

import os
from typing import Callable, List, Optional

from main.nenv.Action import Accept
from main.nenv.Preference import Preference
from main.LLMAgent import LLMAgent as SotaLLMAgent
from main.HybridAgent import LLMAgent as HybridLLMAgent
from main.llm_components.dialogue_generator import DialogueGenerator
from user_experiment.backend.adapters.callback_human_agent import CallbackHumanAgent

PROTOCOL = "ALTERNATING"


class _DialogueWrappedLLMAgent(SotaLLMAgent):
    """SOTA LLMAgent with its bid intact but the natural-language message
    rewritten by HybridAgent's `DialogueGenerator`. Mirrors the dialogue
    layer the Traditional condition uses so the human-facing chat style
    is identical across sessions; the bid-selection process is the only
    independent variable.
    """

    def __init__(
        self,
        *,
        dialogue_generator: DialogueGenerator,
        language: Optional[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dialogue_generator = dialogue_generator
        self._language = language

    def act(self, t, chat_history=None):
        action = super().act(t, chat_history)

        if isinstance(action, Accept):
            # Match HybridAgent's hardcoded accept phrase — no DialogueGenerator
            # call here, identical to the Traditional path.
            action.message = (
                "Teklifinizi kabul ediyorum!" if self._language == "Türkçe"
                else "I accept your offer!"
            )
            return action

        action.message = self._dialogue_generator.generate_message(
            bid=action.bid,
            chat_history=chat_history or [],
        )
        return action


class AgentFactory:
    """Builds the human, LLM, and Traditional negotiation agents."""

    def create_callback_human_agent(
        self,
        profile_path: str,
        input_callback: Callable[[], str],
        output_callback: Callable[[str], None],
        model_name: str,
        session_time: int = 300,
        estimators: List = None,
    ) -> CallbackHumanAgent:
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        preference = Preference(profile_path)
        return CallbackHumanAgent(
            preference=preference,
            session_time=session_time,
            estimators=estimators or [],
            input_callback=input_callback,
            output_callback=output_callback,
            name="Human",
            model_name=model_name,
        )

    def create_llm_agent(
        self,
        profile_path: str,
        model_name: str,
        session_time: int = 300,
        estimators: List = None,
        language: Optional[str] = None,
    ) -> _DialogueWrappedLLMAgent:
        """SOTA LLM picks bids; HybridAgent's DialogueGenerator writes the
        message attached to each bid (identical dialogue layer to the
        Traditional condition)."""
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        if language is not None and language not in ["English", "Türkçe"]:
            raise ValueError(f"Invalid language: {language}. Must be 'English' or 'Türkçe'")

        dialogue_generator = DialogueGenerator(
            profile_json_path=profile_path,
            model_name=model_name,
            language=language,
        )

        return _DialogueWrappedLLMAgent(
            preference=Preference(profile_path),
            session_time=session_time,
            estimators=estimators or [],
            model_name=model_name,
            protocol=PROTOCOL,
            dialogue_generator=dialogue_generator,
            language=language,
        )

    def create_traditional_agent(
        self,
        profile_path: str,
        model_name: str,
        session_time: int = 300,
        estimators: List = None,
        language: Optional[str] = None,
    ) -> HybridLLMAgent:
        """Hybrid (heuristic + dialogue) agent — heuristic picks the bid,
        the same LLM dialogue generator writes the accompanying message."""
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        if language is not None and language not in ["English", "Türkçe"]:
            raise ValueError(f"Invalid language: {language}. Must be 'English' or 'Türkçe'")

        from main.heuristic_strategies.HybridAgent.HybridAgent import HybridAgent

        preference = Preference(profile_path)
        grounding_strategy = HybridAgent(
            preference=preference,
            session_time=session_time,
            estimators=[],
        )
        return HybridLLMAgent(
            preference=preference,
            session_time=session_time,
            estimators=estimators or [],
            model_name=model_name,
            protocol=PROTOCOL,
            language=language,
            grounding_strategy=grounding_strategy,
        )
