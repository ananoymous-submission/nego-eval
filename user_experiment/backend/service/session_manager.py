"""Session lifecycle and threading management."""

import os
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Callable, Any
from main.nenv.Session import Session
from main.nenv.Agent import AbstractAgent


class SessionStatus(Enum):
    """Session lifecycle states."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionContext:
    """Context for managing a single session."""
    session: Session
    thread: Optional[threading.Thread] = None
    input_queue: queue.Queue = field(default_factory=queue.Queue)
    status: SessionStatus = SessionStatus.CREATED
    error: Optional[Exception] = None
    result: Optional[Any] = None


class SessionManager:
    """Manages negotiation session lifecycle and threading."""

    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}

    def create_session(
        self,
        session_id: str,
        agent_a: AbstractAgent,
        agent_b: AbstractAgent,
        log_path: str,
        deadline_time: int,
        deadline_round: Optional[int] = None,
        loggers: list = None
    ) -> Session:
        """
        Create and register a new session.

        Args:
            session_id: Unique session identifier
            agent_a: First agent (typically human)
            agent_b: Second agent (typically LLM)
            log_path: Path to save session log
            deadline_time: Time deadline in seconds
            deadline_round: Round deadline (optional)
            loggers: Additional loggers (optional)

        Returns:
            Created Session instance

        Raises:
            ValueError: If session_id already exists
        """
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        session = Session(
            agentA=agent_a,
            agentB=agent_b,
            path=log_path,
            deadline_time=deadline_time,
            deadline_round=deadline_round,
            loggers=loggers or []
        )

        context = SessionContext(
            session=session,
            thread=None,
            status=SessionStatus.CREATED
        )

        self.sessions[session_id] = context
        return session

    def start_session_async(
        self,
        session_id: str,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Start session in background thread.

        Args:
            session_id: Session to start
            on_complete: Optional callback when session completes successfully
            on_error: Optional callback when session fails

        Raises:
            ValueError: If session doesn't exist or already running
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        context = self.sessions[session_id]

        if context.status == SessionStatus.RUNNING:
            raise ValueError(f"Session {session_id} is already running")

        def run_session():
            try:
                context.status = SessionStatus.RUNNING
                result = context.session.start()
                context.result = result
                context.status = SessionStatus.COMPLETED

                if on_complete:
                    on_complete(result)

            except Exception as e:
                context.status = SessionStatus.FAILED
                context.error = e

                if on_error:
                    on_error(e)
                else:
                    # Re-raise if no error handler (surface error immediately)
                    raise

        thread = threading.Thread(target=run_session, daemon=True)
        context.thread = thread
        thread.start()

    def send_human_input(self, session_id: str, message: str):
        """
        Send human input to unblock callback.

        Args:
            session_id: Session to send input to
            message: User's message

        Raises:
            ValueError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        context = self.sessions[session_id]
        context.input_queue.put(message)

    def get_status(self, session_id: str) -> SessionStatus:
        """
        Get current session status.

        Args:
            session_id: Session to query

        Returns:
            Current session status

        Raises:
            ValueError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        return self.sessions[session_id].status

    def get_error(self, session_id: str) -> Optional[Exception]:
        """
        Get session error if failed.

        Args:
            session_id: Session to query

        Returns:
            Exception if session failed, None otherwise

        Raises:
            ValueError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        return self.sessions[session_id].error

    def get_result(self, session_id: str) -> Optional[Any]:
        """
        Get session result if completed.

        Args:
            session_id: Session to query

        Returns:
            Result if session completed, None otherwise

        Raises:
            ValueError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        return self.sessions[session_id].result

    def cleanup_session(self, session_id: str):
        """
        Remove session from manager.

        Args:
            session_id: Session to remove

        Raises:
            ValueError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        del self.sessions[session_id]
