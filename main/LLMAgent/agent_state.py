from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Turn:
    """One exchanged turn in the negotiation transcript.

    `bid_body` is the rendered issue->value pairs (e.g.
    "issueA: valueA, issueB: valueB"), or "(no bid)" for action types
    without a bid. `message` is the natural-language message attached to
    the action; only opponent turns can carry one in this codebase.
    """
    speaker: str        # "ME" or "OPPONENT"
    t: float
    bid_body: str
    message: Optional[str] = None


@dataclass
class State:
    """Minimal state for the LLM negotiation agent (ALTERNATING-only).

    `negotiation_history` is the full ordered transcript: every offer
    (and any opponent message attached to that offer) in the order they
    were exchanged. Rendered to the LLM as one line per turn:
        [ME @ t=0.13] issueA: valueA, issueB: valueB
        [OPPONENT @ t=0.16] issueA: valueC, issueB: valueB - "this is my final offer"

    `bid_reasoning_log` records every reasoning string the bidding LLM
    produced, paired with the time fraction t at which it was generated.
    Used for post-hoc analysis.
    """

    negotiation_history: List[Turn] = field(default_factory=list)
    bid_reasoning_log: List[Tuple[float, str]] = field(default_factory=list)
