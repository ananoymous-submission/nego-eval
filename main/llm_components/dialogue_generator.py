"""Shared `DialogueGenerator` used by every LLM-driven negotiation agent.

The generator's job is intentionally narrow: take an already-decided bid
and write a short, neutral chat message that **presents** it. It does not
strategise, justify, persuade, or reveal preferences — those concerns
belong to the bid-selection layer above.
"""

import dspy
from typing import List, Optional

from main.nenv.Bid import Bid
from main.nenv.Preference import Preference
from main.llm_components.base_component import Component


def _language_instruction(language: Optional[str]) -> str:
    if language == "English":
        return "\n** CRITICAL **: You MUST respond in English."
    if language == "Türkçe":
        return "\n** CRITICAL **: Türkçe yanıt vermek ZORUNDASIN."
    return ""


class DialogueGenerationSignature(dspy.Signature):
    """Write one short, natural-sounding chat message that PRESENTS the
    given offer in a conversation.

    Rules — follow strictly:
    - Just present the offer. Do NOT argue for it, justify it, persuade,
      or otherwise try to lead the negotiation.
    - Refer only to issues and values that exist in the negotiation domain.
    - One short, plain, conversational sentence. No paragraphs.

    """

    chat_history: List[str] = dspy.InputField(desc="Recent chat for tone context only")
    preference_domain: str = dspy.InputField(desc="Valid issues and values in this negotiation")
    our_offer: str = dspy.InputField(desc="The decided offer, formatted as 'issue: value | ...'")
    message: str = dspy.OutputField(desc="One short sentence that presents the offer")


class DialogueGenerator(Component):
    """Wraps the signature above with a domain-aware preference summary
    and an optional language switch. Reused by both the SOTA LLM agent
    (where the LLM picks the bid) and the Hybrid agent (where a heuristic
    picks the bid) — same dialogue surface, different bid pipelines.
    """

    def __init__(
        self,
        profile_json_path: str,
        model_name: str = None,
        temperature: float = 0.0,
        language: Optional[str] = None,
    ):
        if language is not None and language not in ["English", "Türkçe"]:
            raise ValueError(
                f"Invalid language: {language}. Must be 'English' or 'Türkçe'."
            )

        self.language = language
        self.preference = Preference(profile_json_path)
        self.preference_domain = self._build_preference_domain()

        base_doc = DialogueGenerationSignature.__doc__

        class _InstanceSignature(DialogueGenerationSignature):
            __doc__ = base_doc + _language_instruction(language)

        super().__init__(
            component_name="DialogueGeneration",
            signature=_InstanceSignature,
            model_name=model_name,
            temperature=temperature,
        )

    def generate_message(self, bid: Bid, chat_history: List[str]) -> str:
        """Return one sentence presenting `bid`. Caller wraps it in an Offer."""
        if bid and bid.content:
            offer_str = " | ".join(f"{issue.name}: {value}" for issue, value in bid)
        else:
            offer_str = "No offer yet"

        history_context = chat_history[-12:] if chat_history else []

        prediction = self.forward(
            chat_history=history_context,
            preference_domain=self.preference_domain,
            our_offer=offer_str,
        )
        return prediction.message

    def _build_preference_domain(self) -> str:
        lines = [
            f"{issue.name}: {', '.join(issue.values)}"
            for issue in self.preference.issues
        ]
        return "Valid issues and values:\n" + "\n".join(f"- {line}" for line in lines)
