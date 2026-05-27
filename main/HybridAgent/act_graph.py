from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from main.nenv.Action import Accept, Action, Offer, SubAgreement
from main.nenv.Agent import AbstractAgent
from main.nenv.Bid import Bid
from main.nenv.Preference import Preference

from .agent_state import State
from main.llm_components.dialogue_generator import DialogueGenerator


class ActState(TypedDict, total=False):
    """State for the act graph."""
    time_fraction: float
    agent_state: State
    bid: Optional[Bid]
    action_type: str
    offer_with_message: Optional[Action]


class ActComponents:
    """Container for component instances used by the act graph."""

    def __init__(
        self,
        dialogue_generator: DialogueGenerator,
        grounding_strategy: Optional[AbstractAgent] = None,
        preference: Optional[Preference] = None,
        protocol: str = "ALTERNATING",
        language: Optional[str] = None,
    ):
        self.dialogue_generator = dialogue_generator
        self.grounding_strategy = grounding_strategy
        self.preference = preference
        self.protocol = protocol
        self.language = language


def create_generate_bid_node(components: ActComponents):
    """Create bid generation node using the configured heuristic strategy."""

    def generate_bid_node(state: ActState) -> ActState:
        agent_state = state["agent_state"]
        time_fraction = state["time_fraction"]

        grounding_strategy = components.grounding_strategy
        if grounding_strategy is None:
            raise ValueError("LLMAgent requires a grounding_strategy for bid generation.")
        decide_action = getattr(grounding_strategy, "decide_action_for_turn", None)
        if not callable(decide_action):
            raise TypeError(
                "grounding_strategy must implement decide_action_for_turn(...) for action selection."
            )

        decision = decide_action(
            t=time_fraction,
            protocol=components.protocol,
            locked_agreements=agent_state.locked_agreements,
            issue_scope=agent_state.last_opponent_scope,
            allow_extra_issue=agent_state.allow_scope_expansion,
        )

        # Scope expansion is only allowed immediately after a sub-agreement.
        agent_state.allow_scope_expansion = False

        return {
            **state,
            "bid": decision["bid"],
            "action_type": decision.get("action_type", "offer"),
        }

    return generate_bid_node


def create_dialogue_generation_node(components: ActComponents):
    """Hand the chosen bid to the shared DialogueGenerator, which writes a
    short neutral message that just presents the offer."""

    def dialogue_generation_node(state: ActState) -> ActState:
        agent_state = state["agent_state"]
        bid = state["bid"]
        action_type = state.get("action_type", "offer")

        if action_type == "accept":
            accept_message = "Teklifinizi kabul ediyorum!" if components.language == "Türkçe" else "I accept your offer!"
            return {
                **state,
                "offer_with_message": Accept(bid=bid, message=accept_message),
            }

        message = components.dialogue_generator.generate_message(
            bid=bid,
            chat_history=agent_state.chat_history,
        )

        action = SubAgreement(bid=bid, message=message) if action_type == "subagreement" else Offer(bid=bid, message=message)
        return {
            **state,
            "offer_with_message": action,
        }

    return dialogue_generation_node


def build_act_graph(components: ActComponents) -> StateGraph:
    """Build the act graph: BidSelection -> DialogueGeneration -> END."""

    graph_builder = StateGraph(ActState)

    graph_builder.add_node("BidSelection", create_generate_bid_node(components))
    graph_builder.add_node("DialogueGeneration", create_dialogue_generation_node(components))

    graph_builder.add_edge(START, "BidSelection")
    graph_builder.add_edge("BidSelection", "DialogueGeneration")
    graph_builder.add_edge("DialogueGeneration", END)

    return graph_builder.compile()
