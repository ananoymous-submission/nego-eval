"""
Username entry page.
"""

import gradio as gr
from typing import Tuple, Dict, Any
from user_experiment.backend.database.db_service import DatabaseService


def create_username_page(db_service: DatabaseService) -> Tuple[gr.Column, gr.Button, list, list, callable]:
    """
    Create username entry page.

    Args:
        db_service: DatabaseService instance for database operations

    Returns:
        Tuple of:
        - username_page: gr.Column containing the username UI
        - username_submit_btn: gr.Button for submission
        - username_inputs: List of input components
        - username_outputs: List of output components
        - handle_username_submit: Submit handler function
    """

    with gr.Column(visible=True) as username_page:
        gr.Markdown("## Welcome to the Negotiation Experiment! (Deneyimize Hoş Geldiniz!)")
        gr.Markdown("Please enter a username to begin. (Başlamak için lütfen bir kullanıcı adı girin.)")

        username_input = gr.Textbox(
            label="Username / Kullanıcı Adı",
            placeholder="Enter your username / Kullanıcı adınızı girin",
            max_lines=1
        )

        language_radio = gr.Radio(
            choices=["English", "Türkçe"],
            label="Language / Dil",
            value="English",
            info="Select your preferred language. (Lütfen en hakim olduğunuz dili seçin)"
        )

        username_error = gr.Markdown("", visible=False)
        username_submit_btn = gr.Button("Continue (Devam Et)", variant="primary", size="lg")

    def handle_username_submit(
        username: str,
        language: str,
        state: Dict[str, Any]
    ) -> Tuple[str, gr.update, gr.update, Dict[str, Any]]:
        """
        Validate username and language, then transition to survey page.

        Args:
            username: Username entered by user
            language: Selected language (English or Türkçe)
            state: Current user state

        Returns:
            Tuple of (error_message, username_page_update, survey_page_update, updated_state)
        """
        if not username or not username.strip():
            return (
                "Please enter a username / Lütfen bir kullanıcı adı girin.",
                gr.update(visible=True),
                gr.update(visible=False),
                state
            )

        if not language:
            return (
                "Please select a language / Lütfen en hakim olduğunuz dili seçin.",
                gr.update(visible=True),
                gr.update(visible=False),
                state
            )

        if language not in ["English", "Türkçe"]:
            return (
                f"Invalid language selection: {language}",
                gr.update(visible=True),
                gr.update(visible=False),
                state
            )

        username = username.strip()

        try:
            # Check if user exists
            user = db_service.get_user_by_username(username)

            if user is None:
                # Create new user
                user_id = db_service.create_user(username)
            else:
                user_id = user['user_id']

            # Update state
            state["user_id"] = user_id
            state["username"] = username
            state["language"] = language

            # Success - hide username page, show survey
            return (
                "",
                gr.update(visible=False),  # Hide username
                gr.update(visible=True),   # Show survey
                state
            )

        except Exception as e:
            return (
                f"Database error: {str(e)}",
                gr.update(visible=True),
                gr.update(visible=False),
                state
            )

    # Collect input and output components
    username_inputs = [username_input, language_radio]
    username_outputs = [username_error]

    return (
        username_page,
        username_submit_btn,
        username_inputs,
        username_outputs,
        handle_username_submit
    )
