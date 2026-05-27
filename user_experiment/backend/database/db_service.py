"""Database service for user and session tracking."""

import sqlite3
import os
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager


class DatabaseService:
    """Service for managing SQLite database operations."""

    def __init__(self, db_path: str = "negotiation.db"):
        """
        Initialize database service.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database and tables if they don't exist."""
        schema_path = os.path.join(
            os.path.dirname(__file__),
            "schema.sql"
        )

        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Database schema not found: {schema_path}")

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def create_user(self, username: str) -> int:
        """
        Create a new user.

        Args:
            username: Unique username

        Returns:
            user_id of created user

        Raises:
            sqlite3.IntegrityError: If username already exists
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username) VALUES (?)",
                (username,)
            )
            conn.commit()
            return cursor.lastrowid

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username.

        Args:
            username: Username to look up

        Returns:
            User dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> int:
        """
        Save user survey preferences.

        Args:
            user_id: User ID from users table
            preferences: Dict containing survey responses with keys:
                - strategy_preference
                - risk_tolerance
                - outcome_preference
                - communication_style
                - problem_approach

        Returns:
            preference_id of inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_preferences (
                    user_id,
                    strategy_preference,
                    risk_tolerance,
                    outcome_preference,
                    communication_style,
                    problem_approach
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    preferences["strategy_preference"],
                    preferences["risk_tolerance"],
                    preferences["outcome_preference"],
                    preferences["communication_style"],
                    preferences["problem_approach"]
                )
            )
            conn.commit()
            return cursor.lastrowid

    def create_session(
        self,
        session_id: str,
        user_id: int,
        llm_model_name: str,
        human_profile_path: str,
        llm_profile_path: str,
        log_path: str
    ) -> None:
        """
        Create a new negotiation session record.

        Args:
            session_id: Unique session identifier
            user_id: User ID from users table
            llm_model_name: LLM model name
            human_profile_path: Path to human profile
            llm_profile_path: Path to LLM profile
            log_path: Path to session log file
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO negotiation_sessions
                (session_id, user_id, llm_model_name, human_profile_path,
                 llm_profile_path, status, log_path)
                VALUES (?, ?, ?, ?, ?, 'in_progress', ?)
                """,
                (session_id, user_id, llm_model_name,
                 human_profile_path, llm_profile_path, log_path)
            )
            conn.commit()

    def update_session_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update session status.

        Args:
            session_id: Session to update
            status: New status ('in_progress', 'completed', 'failed')
            error_message: Optional error message if failed
        """
        with self._get_connection() as conn:
            if status in ['completed', 'failed']:
                conn.execute(
                    """
                    UPDATE negotiation_sessions
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE session_id = ?
                    """,
                    (status, datetime.now(), error_message, session_id)
                )
            else:
                conn.execute(
                    """
                    UPDATE negotiation_sessions
                    SET status = ?
                    WHERE session_id = ?
                    """,
                    (status, session_id)
                )
            conn.commit()

    def get_user_sessions(self, user_id: int) -> list[Dict[str, Any]]:
        """
        Get all sessions for a user.

        Args:
            user_id: User ID to query

        Returns:
            List of session dicts
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM negotiation_sessions
                WHERE user_id = ?
                ORDER BY started_at DESC
                """,
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
