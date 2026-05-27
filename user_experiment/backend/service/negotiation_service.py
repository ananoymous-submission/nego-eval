"""Main orchestration service for negotiation sessions.

Provides clean interface between UI and negotiation framework.
"""

import os
import json
from typing import Callable, Optional
from user_experiment.backend.service.session_manager import SessionManager, SessionStatus
from user_experiment.backend.service.agent_factory import AgentFactory
from user_experiment.backend.config.settings import SessionConfig


class NegotiationService:
    """
    Service layer for managing negotiation sessions.

    Provides clean interface between UI and core framework.
    """

    def __init__(self, config: SessionConfig):
        """
        Initialize negotiation service.

        Args:
            config: Session configuration
        """
        self.config = config
        self.session_manager = SessionManager()
        self.agent_factory = AgentFactory()

    def start_session(
        self,
        session_id: str,
        human_profile_path: str,
        llm_profile_path: str,
        llm_model_name: str,
        get_human_input: Callable[[], str],
        on_llm_message: Callable[[str], None],
        agent_type: str,
        language: Optional[str] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ) -> str:
        """
        Start a negotiation session.

        Args:
            session_id: Unique session identifier
            human_profile_path: Path to human preference profile
            llm_profile_path: Path to opponent preference profile
            llm_model_name: LLM model name (used by both LLM and Traditional)
            get_human_input: Callback to get human input (blocking)
            on_llm_message: Callback when opponent sends message
            agent_type: Which AI opponent to spawn — "LLM" (pure-LLM bidder)
                        or "Traditional" (heuristic bidder + LLM dialogue).
            language: Language for opponent dialogue ("English" or "Türkçe").
                      Only consumed by the Traditional agent; ignored by LLM.
            on_complete: Optional callback when session completes
            on_error: Optional callback when session fails

        Returns:
            Session ID

        Raises:
            ValueError: If agent_type/language is invalid.
            FileNotFoundError: If profile paths don't exist.
        """
        if agent_type not in ("LLM", "Traditional"):
            raise ValueError(
                f"Invalid agent_type: {agent_type!r}. Must be 'LLM' or 'Traditional'."
            )
        if language is not None and language not in ["English", "Türkçe"]:
            raise ValueError(f"Invalid language: {language}. Must be 'English' or 'Türkçe'")

        human_agent = self.agent_factory.create_callback_human_agent(
            profile_path=human_profile_path,
            input_callback=get_human_input,
            output_callback=on_llm_message,
            model_name=llm_model_name,
            session_time=self.config.deadline_time,
        )

        if agent_type == "LLM":
            opponent_agent = self.agent_factory.create_llm_agent(
                profile_path=llm_profile_path,
                model_name=llm_model_name,
                session_time=self.config.deadline_time,
                language=language,
            )
        else:
            opponent_agent = self.agent_factory.create_traditional_agent(
                profile_path=llm_profile_path,
                model_name=llm_model_name,
                session_time=self.config.deadline_time,
                language=language,
            )

        # Generate log path
        log_path = self._get_log_path(session_id, llm_model_name)

        self.session_manager.create_session(
            session_id=session_id,
            agent_a=human_agent,
            agent_b=opponent_agent,
            log_path=log_path,
            deadline_time=self.config.deadline_time,
            deadline_round=self.config.deadline_round,
            loggers=[]
        )

        def _on_complete_with_reasoning_dump(result):
            # Persist the LLM-side bid reasoning alongside the xlsx (same
            # `<base>.bidding.json` sidecar shape tournament.py writes).
            # Only the LLM condition has a reasoning log to dump; the
            # Traditional path's bidder is heuristic.
            self._write_bidding_log(opponent_agent, log_path)
            if on_complete is not None:
                on_complete(result)

        self.session_manager.start_session_async(
            session_id=session_id,
            on_complete=_on_complete_with_reasoning_dump,
            on_error=on_error
        )

        return session_id

    def send_human_message(self, session_id: str, message: str):
        """
        Send human message to session (unblocks get_human_input callback).

        Args:
            session_id: Session to send message to
            message: User's message

        Raises:
            ValueError: If session doesn't exist
        """
        self.session_manager.send_human_input(session_id, message)

    def start_negotiation_timer(self, session_id: str):
        """
        Start the deadline timer for a negotiation session.

        This should be called when the user sends their first message, so that the
        countdown timer only begins when actual negotiation starts (not when the user
        is reading instructions on the negotiation page).

        Args:
            session_id: Session to start timer for

        Raises:
            ValueError: If session doesn't exist
        """
        session_context = self.session_manager.sessions.get(session_id)
        if not session_context:
            raise ValueError(f"Session {session_id} not found")

        session_context.session.start_deadline_timer()
        print(f"[NegotiationService] Started deadline timer for session {session_id}")

    def get_session_status(self, session_id: str) -> SessionStatus:
        """
        Get current session status.

        Args:
            session_id: Session to query

        Returns:
            Current session status

        Raises:
            ValueError: If session doesn't exist
        """
        return self.session_manager.get_status(session_id)

    def get_session_error(self, session_id: str) -> Optional[Exception]:
        """
        Get session error if failed.

        Args:
            session_id: Session to query

        Returns:
            Exception if session failed, None otherwise

        Raises:
            ValueError: If session doesn't exist
        """
        return self.session_manager.get_error(session_id)

    def cleanup_session(self, session_id: str):
        """
        Remove session from manager.

        Args:
            session_id: Session to remove

        Raises:
            ValueError: If session doesn't exist
        """
        self.session_manager.cleanup_session(session_id)

    def save_user_profile(
        self,
        username: str,
        language: str,
        domain: str,
        profile_data: dict
    ) -> str:
        """
        Save user-created profile to disk.

        Args:
            username: User identifier
            language: "English" or "Türkçe"
            domain: "holiday" or "resource"
            profile_data: Dict with issueWeights and issues

        Returns:
            Path to saved profile file

        Raises:
            ValueError: If validation fails
        """
        # Validate profile structure
        self._validate_profile(profile_data)

        # Map language to directory
        language_dir = "turkisch" if language == "Türkçe" else "englisch"

        # Create directory path (grouped by language/domain/user for easier LLM agent profile creation later)
        profile_dir = f"data/user_profiles/{language_dir}/{domain}/{username}"
        os.makedirs(profile_dir, exist_ok=True)

        # Full profile path
        profile_path = f"{profile_dir}/profile.json"

        # Write profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)

        print(f"Saved user profile: {profile_path}")
        return profile_path

    def _apply_mixed_inversion(self, values_list: list) -> list:
        """
        Apply Mixed inversion method to a list of values.

        Mixed method rotates values to create partial opposition (not complete reversal).
        - Empty/1-2 elements: Simple reverse
        - Even length: Rotate by half (second half + first half)
        - Odd length: Rotate with middle element preserved in center

        Args:
            values_list: List of numeric values to invert

        Returns:
            Inverted list (same length, rotated order)

        Examples:
            [1, 2, 3, 4] -> [3, 4, 1, 2]
            [1, 2, 3, 4, 5] -> [4, 5, 3, 1, 2]
            [1, 2] -> [2, 1]
        """
        if len(values_list) == 0:
            return values_list

        if len(values_list) <= 2:
            return values_list[::-1]

        mid = len(values_list) // 2

        if len(values_list) % 2 == 0:
            # Even length: rotate by half
            return values_list[mid:] + values_list[:mid]
        else:
            # Odd length: rotate with middle element in center
            left = values_list[:mid]
            right = values_list[mid+1:]
            middle = [values_list[mid]]
            return right + middle + left

    def generate_inverted_profile(
            self,
            username: str,
            language: str,
            domain: str,
            human_profile_data: dict
        ) -> str:
            """
            Generate inverted opponent profile using Mixed inversion method.
            
            CRITICAL UPDATE: Sorts items by utility before inversion to ensure 
            deterministic 'Mixed' behavior (e.g. always rotating Best -> Medium).
            """
            
            # --- 1. ISSUE WEIGHTS ---
            # Agent weights are fixed: same as human but Accommodation <-> Activities swapped.
            AGENT_ISSUE_WEIGHTS = {
                "englisch": {
                    "Destination": 0.3,
                    "Season": 0.25,
                    "Activities": 0.2,
                    "Accommodation": 0.15,
                    "Transportation": 0.1,
                },
                "turkisch": {
                    "Destinasyon": 0.3,
                    "Mevsim": 0.25,
                    "Aktivite": 0.2,
                    "Konaklama": 0.15,
                    "Ulaşım": 0.1,
                },
            }
            language_dir = "turkisch" if language == "Türkçe" else "englisch"
            inverted_issue_weights = AGENT_ISSUE_WEIGHTS[language_dir]

            # --- 2. INVERT VALUE UTILITIES (per issue) ---
            # Destination and Season are inverted fully (reversed); all others use Mixed.
            fully_inverted_issues = {"Destination", "Season", "Destinasyon", "Mevsim"}

            inverted_issues = {}
            for issue_name in human_profile_data["issueWeights"].keys():
                # Get values for this issue
                raw_values = human_profile_data["issues"][issue_name].items()

                # SORT by utility (Descending) -> [Best, ..., Worst]
                sorted_value_items = sorted(raw_values, key=lambda x: x[1], reverse=True)

                sorted_value_names = [item[0] for item in sorted_value_items]
                sorted_utilities = [item[1] for item in sorted_value_items]

                if issue_name in fully_inverted_issues:
                    inverted_utilities_list = sorted_utilities[::-1]
                else:
                    inverted_utilities_list = self._apply_mixed_inversion(sorted_utilities)

                inverted_issues[issue_name] = {
                    name: inverted_utilities_list[i]
                    for i, name in enumerate(sorted_value_names)
                }

            # --- BUILD INVERTED PROFILE ---
            inverted_profile = {
                "reservationValue": human_profile_data.get("reservationValue", 0),
                "issueWeights": inverted_issue_weights,
                "issues": inverted_issues
            }

            # VALIDATE
            self._validate_profile(inverted_profile)

            # SAVE TO FILE
            language_dir = "turkisch" if language == "Türkçe" else "englisch"
            profile_dir = f"data/user_profiles/{language_dir}/{domain}/{username}"
            os.makedirs(profile_dir, exist_ok=True) # Ensure dir exists
            inverted_profile_path = f"{profile_dir}/profileB.json"

            with open(inverted_profile_path, 'w', encoding='utf-8') as f:
                json.dump(inverted_profile, f, indent=2, ensure_ascii=False)

            print(f"Saved sorted & inverted LLM profile: {inverted_profile_path}")
            return inverted_profile_path

    def _validate_profile(self, profile_data: dict):
        """
        Validate profile structure and constraints.

        Args:
            profile_data: Profile data to validate

        Raises:
            ValueError: If validation fails

        Returns:
            True if valid
        """
        # Check required keys
        if "issueWeights" not in profile_data:
            raise ValueError("Missing issueWeights")
        if "issues" not in profile_data:
            raise ValueError("Missing issues")

        # Validate issue weights sum to 1.0 (with tolerance)
        weight_sum = sum(profile_data["issueWeights"].values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Issue weights must sum to 1.0, got {weight_sum}")

        # Validate value utilities are between 0 and 1
        for issue_name, values in profile_data["issues"].items():
            for value_name, utility in values.items():
                if not (0.0 <= utility <= 1.0):
                    raise ValueError(
                        f"Utility for {issue_name}/{value_name} must be between 0 and 1, got {utility}"
                    )

        return True

    def _get_log_path(self, session_id: str, model_name: str) -> str:
        """Log path is `<log_dir>/<session_id>.xlsx`. The session_id already
        encodes username/agent_type/datetime — model_name is ignored."""
        return f"{self.config.log_directory}/{session_id}.xlsx"

    @staticmethod
    def _write_bidding_log(agent, log_path_xlsx: str) -> None:
        """Write the LLM-side `state.bid_reasoning_log` to a
        `<base>.bidding.json` sidecar (same shape as tournament.py's
        `write_llm_reasoning_log`). No-op for agents that don't expose a
        reasoning log — i.e. the Traditional path's heuristic bidder.
        """
        state = getattr(agent, "state", None)
        bid_log = getattr(state, "bid_reasoning_log", None) if state is not None else None
        if not bid_log:
            return

        sidecar = log_path_xlsx.rsplit(".", 1)[0] + ".bidding.json"
        payload = {
            "agent_name": getattr(agent, "name", agent.__class__.__name__),
            "model_name": getattr(agent, "model_name", None),
            "bids": [{"t": t, "reasoning": reasoning} for (t, reasoning) in bid_log],
        }
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
