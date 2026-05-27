"""Tournament configuration.

This file is the single source of truth for what plays in the tournament.

- PURE_LLMS:              models that play as LLMAgent (LLM picks bids directly).
                          One LLMAgent participant per model. Empty list disables
                          all LLM-side sessions.
- TRADITIONAL_OPPONENTS:  display-name -> AbstractAgent subclass. Each entry
                          plays as a TRAD side against every LLM and every other
                          TRAD. Empty dict disables all TRAD-side sessions.
- DOMAINS:                domain folder names under main/domains/.
- DEADLINE_ROUND:         maximum rounds per session before timeout.
- DEADLINE_TIME:          wall-clock seconds per session (None = round-only).

The tournament matrix is built from whatever sets are non-empty. Set
PURE_LLMS=[] to run only the traditional baseline.
"""

from typing import Dict, List, Optional, Type

from main.nenv.Agent import AbstractAgent

from agents.HybridAgent.HybridAgent import HybridAgent
from agents.boulware.Boulware import BoulwareAgent
from agents.conceder.Conceder import ConcederAgent
from agents.NiceTitForTat.NiceTitForTat import NiceTitForTat
from agents.ParsCat.ParsCat import ParsCatAgent
from agents.SAGA.SAGAAgent import SAGAAgent
from agents.HardHeaded.KLH import HardHeaded

PURE_LLMS: List[str] = [
    "openai/gpt-4o-2024-08-06",
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/openai/gpt-5.5",
    "openrouter/~google/gemini-pro-latest",
    "openrouter/x-ai/grok-4.3",
]

TRADITIONAL_OPPONENTS: Dict[str, Type[AbstractAgent]] = {
    "Boulware": BoulwareAgent,
    "Conceder": ConcederAgent,
    "NiceTitForTat": NiceTitForTat,
    "SAGA": SAGAAgent,
    "HardHeaded": HardHeaded,
    "HybridAgent": HybridAgent,
}

DOMAINS: List[int] = [
    12, 44, 5, 37, 10, 42, 7, 39
]

DEADLINE_ROUND: int = 50
DEADLINE_TIME: Optional[int] = None


# =============================================================================
# Robustness sweep (consumed by robustness_tournament.py — separate runner)
# =============================================================================
# Each adversary wraps a fixed classical bidder + persona-flavoured dialogue.
# Only the dialogue varies across personas; bidding stays identical so the
# difference in victim utility is attributable to the dialogue channel.
#
# ADVERSARIES: display-name -> persona name (must be one of
#              main.adversaries.persona_dialogue.ALLOWED_PERSONAS).
# ROBUSTNESS_ADVERSARY_BIDDER: classical agent class wrapped by every adversary.

ADVERSARIES: Dict[str, str] = {
    "Adversary-threat":      "threat",
    "Adversary-emotional":   "emotional",
    "Adversary-positive":    "positive_manipulation",
    "Adversary-constraints": "false_constraints",
    "Adversary-urgency":     "fake_urgency",
    "Adversary-injection":   "prompt_injection",
}

ROBUSTNESS_ADVERSARY_BIDDER: Type[AbstractAgent] = BoulwareAgent
