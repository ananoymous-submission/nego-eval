"""
Human Act Graph

Implements the HumanAgent's act() method as a LangGraph pipeline for visualization and tracking.
Focuses on message classification and validation.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List
from main.nenv.Action import Action
from main.nenv.MessageType import MessageType
from .MessageClassifier import MessageClassifier


class HumanActState(TypedDict, total=False):
    """State for the human act graph."""
    # Core pipeline data
    message: str                    # User input message
    message_type: MessageType        # Classification result
    confidence: float               # Classification confidence
    action: Optional[Action]        # Final action to return

    # Context (passed in, not modified)
    chat_history: List[str]         # Chat context

    # Components
    classifier: MessageClassifier   # Message classifier instance

    # Control flow
    error_message: Optional[str]    # For user feedback
    needs_retry: bool               # Whether to retry input


def user_input_node(state: HumanActState) -> HumanActState:
    """Get user input from terminal."""
    while True:
        try:
            message = input("Enter your response: ").strip()
            if message:
                return {**state, "message": message, "needs_retry": False}
            else:
                print("Please enter a message.")
        except KeyboardInterrupt:
            print("\nNegotiation cancelled by user.")
            raise
        finally:
            print()



def classify_message_node(state: HumanActState) -> HumanActState:
    """Classify the message type using MessageClassifier."""
    classifier = state["classifier"]
    chat_history = state.get("chat_history", [])
    message = state["message"]

    classification_result = classifier.classify(message, chat_history)

    return {
        **state,
        "message_type": classification_result["type"],
        "confidence": classification_result["confidence"]
    }


def build_action_node(state: HumanActState) -> HumanActState:
    """Build the appropriate Action object based on message type."""
    message_type = state["message_type"]
    message = state["message"]

    if message_type == MessageType.ACCEPT:
        return {**state, "action": "ACCEPT_PLACEHOLDER", "message": message}

    elif message_type == MessageType.OFFER:
        return {
            **state,
            "error_message": "Offers must be submitted through the offer panel.",
            "needs_retry": True,
            "action": None
        }

    else:
        raise ValueError(f"Unrecognized message type: {message_type}")


def route_final_action(state: HumanActState) -> str:
    """Route to retry or end based on whether we have a valid action."""
    if state.get("needs_retry", False):
        # Print error message and retry
        error_msg = state.get("error_message")
        if error_msg:
            print(error_msg)
        return "user_input"
    else:
        return END


def build_human_act_graph() -> StateGraph:
    """Build the human act graph for processing user input into actions."""

    # Build graph
    graph_builder = StateGraph(HumanActState)

    # Add nodes
    graph_builder.add_node("user_input", user_input_node)
    graph_builder.add_node("classify_message", classify_message_node)
    graph_builder.add_node("build_action", build_action_node)

    # Start with user input
    graph_builder.add_edge(START, "user_input")

    # Always classify after getting input
    graph_builder.add_edge("user_input", "classify_message")
    graph_builder.add_edge("classify_message", "build_action")

    # After building action, decide to retry or end
    graph_builder.add_conditional_edges(
        "build_action",
        route_final_action,
        {
            "user_input": "user_input",
            END: END
        }
    )

    # Compile graph
    return graph_builder.compile()
