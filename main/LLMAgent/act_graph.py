from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from main.nenv.Action import Accept, Action, Offer
from main.nenv.Bid import Bid
from main.nenv.Preference import Preference

from .agent_state import State
from .components.bidding_strategy import BiddingStrategy


class ActState(TypedDict, total=False):
    time_fraction: float
    agent_state: State
    last_opponent_bid: Optional[Bid]
    bid: Optional[Bid]
    action_type: str  # "offer" | "accept"
    offer_with_message: Optional[Action]


class ActComponents:

    def __init__(
        self,
        bidding_strategy: BiddingStrategy,
        preference: Preference,
    ):
        self.bidding_strategy = bidding_strategy
        self.preference = preference


def create_bid_selection_node(components: ActComponents):

    def bid_selection_node(state: ActState) -> ActState:
        agent_state = state["agent_state"]
        t = state["time_fraction"]
        opp_last = state.get("last_opponent_bid")

        bid, reasoning = components.bidding_strategy.generate_bid(
            negotiation_history=agent_state.negotiation_history,
            time_fraction=t,
        )

        agent_state.bid_reasoning_log.append((float(t), reasoning))

        # AC_Next: accept opponent's last bid if it gives us at least as much
        # utility as the bid we are about to send.
        if opp_last is not None and opp_last.utility >= bid.utility:
            return {
                **state,
                "bid": opp_last,
                "action_type": "accept",
                "offer_with_message": Accept(bid=opp_last),
            }

        return {
            **state,
            "bid": bid,
            "action_type": "offer",
            "offer_with_message": Offer(bid=bid),
        }

    return bid_selection_node


def build_act_graph(components: ActComponents) -> StateGraph:
    graph_builder = StateGraph(ActState)

    graph_builder.add_node("BidSelection", create_bid_selection_node(components))

    graph_builder.add_edge(START, "BidSelection")
    graph_builder.add_edge("BidSelection", END)

    return graph_builder.compile()
