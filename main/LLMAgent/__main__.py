from typing import Optional, List

from main.nenv.Action import Action
from main.nenv.Bid import Bid
from main.nenv.Preference import Preference
from main.nenv.Agent import AbstractAgent
from main.opponent_models.AbstractOpponentModel import AbstractOpponentModel

from .agent_state import State, Turn
from .components.bidding_strategy import BiddingStrategy
from .act_graph import build_act_graph, ActState, ActComponents


def _bid_body(bid: Optional[Bid]) -> str:
    if bid is None or not getattr(bid, "content", None):
        return "(no bid)"
    return ", ".join(
        f"{issue.name if hasattr(issue, 'name') else str(issue)}: {value}"
        for issue, value in bid
    )


class LLMAgent(AbstractAgent):
    """LLM-only negotiation agent (ALTERNATING protocol).

    Each turn the LLM sees the full offer history (with speaker + time),
    the opponent's message history (when dialogue_aware), its own preference,
    and the current time fraction, then emits the next bid directly. AC_Next
    is applied on top: if the opponent's last offer is at least as good for us
    as the bid the LLM just generated, we accept.

    `dialogue_aware` toggles whether the LLM is shown opponent messages. When
    False (dialogue-blind floor for robustness ablation), the LLM never sees
    opponent text even though we still record it in state for analysis.
    """

    def __init__(
        self,
        preference: Preference,
        session_time: int = 180,
        estimators: List[AbstractOpponentModel] = None,
        model_name: str = None,
        protocol: str = None,
        dialogue_aware: bool = True,
    ):
        super().__init__(preference, session_time, estimators or [])

        if protocol != "ALTERNATING":
            raise ValueError(
                f"LLMAgent only supports protocol='ALTERNATING' (got {protocol!r})."
            )

        self.protocol = protocol
        self.model_name = model_name
        self.dialogue_aware = dialogue_aware
        self.state = State()

        self.bidding_strategy = BiddingStrategy(
            preference=preference,
            profile_json_path=preference.profile_json_path,
            model_name=model_name,
            dialogue_aware=dialogue_aware,
        )

        self.act_components = ActComponents(
            bidding_strategy=self.bidding_strategy,
            preference=preference,
        )
        self.act_graph = build_act_graph(self.act_components)

    @property
    def name(self) -> str:
        return "LLMAgent"

    def initiate(self, opponent_name: Optional[str]):
        return None

    def receive_offer(self, bid, t: float):
        return None

    def receive_action(self, action, t: float, chat_history: List[str] = None):
        super().receive_action(action, t, chat_history)
        bid = getattr(action, "bid", None)
        # Opponent message (if any) is folded into the same turn record. Stored
        # regardless of dialogue_aware so analysis logs stay complete; the
        # BiddingStrategy strips it from the LLM view when dialogue_aware=False.
        message = getattr(action, "message", None) or None
        self.state.negotiation_history.append(
            Turn(speaker="OPPONENT", t=t, bid_body=_bid_body(bid), message=message)
        )

    def act(self, t: float, chat_history: List[str] = None) -> Action:
        last_opp = self.last_received_bids[-1] if self.last_received_bids else None
        initial_state = ActState(
            time_fraction=t,
            agent_state=self.state,
            last_opponent_bid=last_opp,
        )

        result = self.act_graph.invoke(initial_state, {"run_name": f"{self.model_name} LLMAgent Act"})
        action = result["offer_with_message"]

        bid = getattr(action, "bid", None)
        self.state.negotiation_history.append(
            Turn(speaker="ME", t=t, bid_body=_bid_body(bid), message=None)
        )

        return action

    def visualize(self, output_dir: str):
        self.act_graph.get_graph().draw_mermaid_png(
            output_file_path=f"{output_dir}/act_graph.png"
        )
