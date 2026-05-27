from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from main.nenv.Bid import Bid
from main.nenv.Action import Action
from main.nenv.Agent import AbstractAgent
from main.nenv.MessageType import MessageType


class ReceiveState(TypedDict):
    """State for the receive graph."""
    received_action: Action
    received_bid: Optional[Bid]
    time_fraction: float
    message_type: MessageType


class ReceiveComponents:
    """Container for receive graph components."""
    def __init__(self, grounding_strategy: Optional[AbstractAgent] = None):
        self.grounding_strategy = grounding_strategy


def create_grounding_update_node(components: ReceiveComponents):
    """Update grounding strategy's opponent model with received bid."""

    def grounding_update_node(state: ReceiveState) -> ReceiveState:
        received_bid = state.get("received_bid")
        time_fraction = state["time_fraction"]

        if received_bid is not None and components.grounding_strategy:
            components.grounding_strategy.receive_offer(received_bid, time_fraction)

        return {}

    return grounding_update_node


def build_receive_graph(components: ReceiveComponents) -> StateGraph:
    """Build the receive graph (grounding update only)."""

    graph_builder = StateGraph(ReceiveState)

    if components.grounding_strategy:
        graph_builder.add_node("GroundingUpdate", create_grounding_update_node(components))
        graph_builder.add_edge(START, "GroundingUpdate")
        graph_builder.add_edge("GroundingUpdate", END)
    else:
        graph_builder.add_edge(START, END)

    return graph_builder.compile()
