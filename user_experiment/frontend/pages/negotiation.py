"""
Negotiation interface page.
"""

import gradio as gr
from typing import Tuple, Callable
from user_experiment.backend.config.settings import AppConfig
from user_experiment.frontend.components.preferences_display import create_preferences_display
from user_experiment.frontend.components.timer import create_timer_display

def create_negotiation_page(
    chat_fn: Callable,
    config: AppConfig,
    user_state: gr.State,
    page_idx: int = 1
) -> Tuple[gr.Column, gr.ChatInterface, gr.Markdown, gr.Button, gr.Markdown, gr.HTML, gr.HTML, gr.HTML, gr.Row, gr.HTML, gr.Textbox, gr.Button, gr.Button, gr.Textbox, gr.Textbox]:
    """
    Create negotiation chat interface page.

    Args:
        chat_fn: Chat handler function
        config: Application configuration
        user_state: Gradio State object for user session
        page_idx: 1 or 2 — used to scope offer panel JS/IDs

    Returns:
        Tuple of (negotiation_page, chat_interface, progress_text, continue_btn, status_text,
                  timer_display, preferences_display, instructions_html, chat_row,
                  offer_panel_html, offer_textbox, accept_bridge_button,
                  send_offer_bridge_button, send_offer_message_textbox,
                  locked_offer_textbox)
    """

    with gr.Column(visible=False) as negotiation_page:
        # Progress indicator
        progress_text = gr.Markdown("", visible=True)

        # Negotiation rules / how-to-negotiate display
        instructions_html = gr.HTML("", visible=True, elem_classes=["instructions-info"])

        # Timer display (15 minutes = 900 seconds)
        # Content will be updated by server-side timer
        timer_display = create_timer_display()

        # Status text (shows completion message)
        status_text = gr.Markdown("", visible=False)

        # Continue button (hidden initially)
        continue_btn = gr.Button(
            "Continue (Devam Et)",
            variant="primary",
            size="lg",
            visible=False
        )

        with gr.Row(visible=True) as chat_row:
            # Left column: Chat interface + offer panel
            with gr.Column(scale=2):
                # Hidden textbox stores current offer JSON selection
                offer_textbox = gr.Textbox(
                    value="{}",
                    visible="hidden",
                    elem_id=f"offer_tb_{page_idx}",
                    elem_classes=["offer-bridge-hidden"]
                )
                # Hidden bridge button for direct Accept action (bypasses chat send button)
                accept_bridge_button = gr.Button(
                    value="accept-bridge",
                    visible=True,
                    elem_id=f"accept_bridge_{page_idx}",
                    elem_classes=["accept-bridge-hidden"]
                )
                send_offer_bridge_button = gr.Button(
                    value="send-offer-bridge",
                    visible=True,
                    elem_id=f"send_offer_bridge_{page_idx}",
                    elem_classes=["accept-bridge-hidden"]
                )
                send_offer_message_textbox = gr.Textbox(
                    value="",
                    visible="hidden",
                    elem_id=f"send_offer_msg_{page_idx}",
                    elem_classes=["offer-bridge-hidden"]
                )
                locked_offer_textbox = gr.Textbox(
                    value="{}",
                    visible="hidden",
                    elem_id=f"offer_lock_tb_{page_idx}",
                    elem_classes=["offer-bridge-hidden"]
                )
                chat_interface = gr.ChatInterface(
                    fn=chat_fn,
                    type="messages",
                    chatbot=gr.Chatbot(height=700, type="messages"),
                    textbox=gr.Textbox(
                        placeholder="Type your message...",
                        elem_id=f"chat_input_{page_idx}",
                        container=False,
                        scale=7
                    ),
                    submit_btn=False,
                    additional_inputs=[user_state, offer_textbox, locked_offer_textbox],
                    additional_outputs=[offer_textbox, locked_offer_textbox],
                    concurrency_limit=20
                )

                # Offer panel (populated at session init)
                offer_panel_html = gr.HTML("")

            # Right column: Preferences display (initially empty, updated dynamically)
            with gr.Column(scale=1):
                with gr.Accordion("Your Preferences (Tercih Profiliniz)", open=True):
                    preferences_display = create_preferences_display()

    return (
        negotiation_page,
        chat_interface,
        progress_text,
        continue_btn,
        status_text,
        timer_display,
        preferences_display,
        instructions_html,
        chat_row,
        offer_panel_html,
        offer_textbox,
        accept_bridge_button,
        send_offer_bridge_button,
        send_offer_message_textbox,
        locked_offer_textbox
    )
