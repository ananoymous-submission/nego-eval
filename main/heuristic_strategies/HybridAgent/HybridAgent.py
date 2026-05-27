"""Classical Hybrid bidder (Time-Based + Behavior-Based) for the
ALTERNATING-only experiment. Replaces the older partial-protocol-aware
`HybridAgentWithOppModel` whose subagreement / locked-issue plumbing
introduced failure modes that no longer apply now that the app runs on
ALTERNATING only.

Implements the `decide_action_for_turn(...)` interface that
`main.HybridAgent.act_graph` expects, so it slots directly into the
existing dialogue-aware HybridLLMAgent as its `grounding_strategy`.
"""

from typing import Dict, List, Optional

from main.nenv.Action import Action, Offer
from main.nenv.Agent import AbstractAgent
from main.nenv.Bid import Bid


class HybridAgent(AbstractAgent):
    """Time-Based + Behavior-Based bidder. [Keskin2021]_"""

    p0: float  #: Initial utility
    p1: float  #: Concession ratio
    p2: float  #: Final utility
    p3: float  #: Empathy score

    # Window for the Behavior-Based component
    W = {
        1: [1],
        2: [0.25, 0.75],
        3: [0.11, 0.22, 0.66],
        4: [0.05, 0.15, 0.3, 0.5],
    }
    my_last_bids: List[Bid]

    @property
    def name(self) -> str:
        return "Hybrid"

    def initiate(self, opponent_name: Optional[str]):
        self.p0 = 1.0
        self.p1 = 0.8
        self.p2 = max(0.6, self.preference.reservation_value)
        self.p3 = 0.5
        self.my_last_bids = []

    def time_based(self, t: float) -> float:
        return (1 - t) * (1 - t) * self.p0 + 2 * (1 - t) * t * self.p1 + t * t * self.p2

    def behaviour_based(self, t: float) -> float:
        diff = [
            self.last_received_bids[i + 1].utility - self.last_received_bids[i].utility
            for i in range(len(self.last_received_bids) - 1)
        ]
        if len(diff) > len(self.W):
            diff = diff[-len(self.W):]
        delta = sum(u * w for u, w in zip(diff, self.W[len(diff)]))
        return self.my_last_bids[-1].utility - (self.p3 + self.p3 * t) * delta

    def receive_offer(self, bid: Bid, t: float):
        # Opponent bids are already tracked by the base via `receive_bid`;
        # we don't maintain an opponent model so nothing extra to do.
        pass

    def _calculate_target_utility(self, t: float) -> float:
        target = self.time_based(t)
        # Need at least 3 received bids to compute the behaviour-based delta.
        if len(self.last_received_bids) > 2 and self.my_last_bids:
            target = (1.0 - t * t) * self.behaviour_based(t) + t * t * target
        return max(target, self.preference.reservation_value)

    def _can_accept(self) -> bool:
        return len(self.last_received_bids) > 0

    def act(self, t: float) -> Action:
        target_utility = self._calculate_target_utility(t)
        if self._can_accept() and target_utility <= self.last_received_bids[-1].utility:
            # Subclasses may not have a preset `accept_action`; fall back to
            # a fresh Accept on the opponent's last bid. Same shape as the
            # `accept_action` shortcut on the base.
            from main.nenv.Action import Accept
            return Accept(bid=self.last_received_bids[-1])

        bid = self.preference.get_bid_at(target_utility)
        self.my_last_bids.append(bid)
        return Offer(bid)

    def decide_action_for_turn(
        self,
        t: float,
        protocol: str,
        locked_agreements: Optional[Dict[str, str]] = None,
        issue_scope: Optional[List[str]] = None,
        allow_extra_issue: bool = False,
    ) -> Dict[str, object]:
        """ALTERNATING-only action selection. The legacy partial-protocol
        kwargs are accepted for interface compatibility with the existing
        `act_graph` call site but are not used.
        """
        if protocol != "ALTERNATING":
            raise ValueError(
                f"HybridAgent only supports protocol='ALTERNATING' (got {protocol!r})."
            )

        target_utility = self._calculate_target_utility(t)

        if self._can_accept() and target_utility <= self.last_received_bids[-1].utility:
            return {
                "action_type": "accept",
                "bid": self.last_received_bids[-1],
                "target_utility": target_utility,
                "reasoning": (
                    f"Opponent's last offer (u={self.last_received_bids[-1].utility:.2f}) "
                    f"meets the current target utility ({target_utility:.2f})."
                ),
            }

        bid = self.preference.get_bid_at(target_utility)
        self.my_last_bids.append(bid)
        return {
            "action_type": "offer",
            "bid": bid,
            "target_utility": target_utility,
            "reasoning": (
                f"Time-based + behavior-based target utility {target_utility:.2f} at t={t:.2f}; "
                f"selected closest bid (u={bid.utility:.2f})."
            ),
        }
