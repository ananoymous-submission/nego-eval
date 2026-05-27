from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class State:
    """Minimal state for the LLM negotiation agent."""

    chat_history: List[str] = field(default_factory=list)
    locked_agreements: Dict[str, str] = field(default_factory=dict)  # issue_name -> locked value
    last_opponent_scope: List[str] = field(default_factory=list)
    allow_scope_expansion: bool = False
