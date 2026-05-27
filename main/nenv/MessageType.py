"""
Message Type Enumeration

Defines the types of negotiation messages that can be exchanged between agents.
Corresponds directly to Action subclasses for consistent protocol handling.
"""

from enum import Enum


class MessageType(Enum):
    """
    Enum for negotiation message types.

    Each type corresponds to specific negotiation semantics:
    - OFFER: Proposes a bid for negotiation (creates Offer action)
    - ACCEPT: Accepts the opponent's current offer (creates Accept action)
    - SUB_AGREEMENT: Accepts a partial offer, locking those issues (creates SubAgreement action)

    Values align with Action classes in main.nenv.Action for consistency.
    """
    OFFER = "offer"
    ACCEPT = "accept"
    SUB_AGREEMENT = "sub_agreement"