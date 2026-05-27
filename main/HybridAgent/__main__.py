from typing import Optional, List

from main.nenv.Action import *
from main.nenv.Preference import Preference
from main.nenv.Agent import AbstractAgent
from main.opponent_models.AbstractOpponentModel import AbstractOpponentModel
from main.heuristic_strategies.HybridAgent.HybridAgent import HybridAgent as _DefaultHybrid

from .agent_state import State
from main.llm_components.dialogue_generator import DialogueGenerator
from .act_graph import build_act_graph, ActState, ActComponents


def _format_history_bid(bid) -> str:
    if bid is None or not getattr(bid, "content", None):
        return ""
    return ", ".join(f"{issue.name if hasattr(issue, 'name') else str(issue)}: {value}" for issue, value in bid)


def _format_history_entry(speaker: str, action, t: float, message_override: Optional[str] = None) -> str:
    action_name = action.__class__.__name__
    message_text = message_override if message_override is not None else getattr(action, "message", "")
    parts = [f"SPEAKER: {speaker}", f"ACTION: {action_name}"]

    bid_text = _format_history_bid(getattr(action, "bid", None))
    if bid_text:
        parts.append(f"BID: {bid_text}")
    if message_text:
        parts.append(f"MESSAGE: {message_text}")

    parts.append(f"t={t:.2f}")
    return "[" + " | ".join(parts) + "]"


class LLMAgent(AbstractAgent):

    def __init__(
        self,
        preference: Preference,
        session_time: int = 180,
        estimators: List[AbstractOpponentModel] = None,
        model_name: str = None,
        protocol: str = None,
        language: Optional[str] = None,
        grounding_strategy: Optional[AbstractAgent] = None,
    ):
        super().__init__(preference, session_time, estimators or [])

        # Validate protocol (required)
        if protocol is None:
            raise ValueError("protocol parameter is required")
        if protocol not in ["ALTERNATING", "PARTIAL"]:
            raise ValueError(f"Invalid protocol: {protocol}. Must be ALTERNATING or PARTIAL")

        # Validate language (optional, but if provided must be valid)
        if language is not None and language not in ["English", "Türkçe"]:
            raise ValueError(f"Invalid language: {language}. Must be 'English' or 'Türkçe'")

        self.protocol = protocol
        self.language = language
        self.model_name = model_name
        self.state = State()
        self.grounding_strategy = grounding_strategy or _DefaultHybrid(
            preference=preference,
            session_time=session_time,
            estimators=[],
        )

        self.dialogue_generator = DialogueGenerator(
            profile_json_path=preference.profile_json_path,
            model_name=model_name,
            language=language,
        )

        # Build act graph only (receive-side modeling is disabled for latency).
        # Bid selection is fully heuristic; the LLM only handles dialogue.
        self.act_components = ActComponents(
            dialogue_generator=self.dialogue_generator,
            grounding_strategy=self.grounding_strategy,
            preference=preference,
            protocol=protocol,
            language=language,
        )
        self.act_graph = build_act_graph(self.act_components)

    @property
    def name(self) -> str:
        return "LLMAgent"

    def initiate(self, opponent_name: Optional[str]):
        if self.grounding_strategy:
            self.grounding_strategy.initiate(opponent_name)

    def receive_offer(self, bid, t: float):
        """Required by AbstractAgent; receive_action handles full synchronization."""
        return None

    def receive_action(self, action, t: float, chat_history: List[str] = None):
        """Process received action from opponent (modeling disabled for latency)."""
        # Treat SubAgreement as a received bid as well, so histories/models stay aligned.
        super().receive_action(action, t, chat_history)

        # Keep lightweight chat history context.
        if chat_history is not None and len(chat_history) > 0:
            opponent_message = chat_history[-1]
            self.state.chat_history = self.state.chat_history + [
                _format_history_entry("OPPONENT", action, t, message_override=opponent_message)
            ]

        # If opponent accepted a sub-agreement, lock those issues
        if isinstance(action, SubAgreement) and action.bid is not None:
            for issue in action.bid.content.keys():
                issue_name = issue.name if hasattr(issue, "name") else str(issue)
                self.state.locked_agreements[issue_name] = action.bid[issue]
            self.state.allow_scope_expansion = True
        else:
            self.state.allow_scope_expansion = False

        # Track scope from the latest bid-carrying action (including sub-agreements).
        # Keep locked issues in scope as well so partial selection constraints stay consistent.
        received_bid = getattr(action, "bid", None)
        if received_bid is not None:
            scope = [
                issue.name if hasattr(issue, "name") else str(issue)
                for issue in received_bid.content.keys()
            ]
            for locked_issue in self.state.locked_agreements.keys():
                if locked_issue not in scope:
                    scope.append(locked_issue)
            self.state.last_opponent_scope = scope

        # Keep grounding strategy state in sync with all bid-carrying actions,
        # including SubAgreement (same accepted bid re-shared by opponent).
        if received_bid is not None and self.grounding_strategy:
            self.grounding_strategy.receive_bid(received_bid, t)

    def act(self, t: float, chat_history: List[str] = None) -> Action:
        initial_state = ActState(
            time_fraction=t,
            agent_state=self.state
        )

        result = self.act_graph.invoke(initial_state, {"run_name": f"{self.model_name} Act Graph"})

        action = result["offer_with_message"]

        # Persist locks when we ourselves send a sub-agreement.
        if isinstance(action, SubAgreement) and action.bid is not None:
            for issue in action.bid.content.keys():
                issue_name = issue.name if hasattr(issue, "name") else str(issue)
                self.state.locked_agreements[issue_name] = action.bid[issue]

        # Track our own message in enriched chat history
        if hasattr(action, 'message') and action.message:
            self.state.chat_history = self.state.chat_history + [
                _format_history_entry("ME", action, t)
            ]

        return action

    def visualize(self, output_dir: str):
        """Generate visualization of the act graph."""
        self.act_graph.get_graph().draw_mermaid_png(
            output_file_path=f"{output_dir}/act_graph.png"
        )
