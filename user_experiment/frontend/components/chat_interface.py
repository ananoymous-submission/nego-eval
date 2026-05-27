"""Chat interface component for negotiation UI."""

from typing import Callable
import gradio as gr


def create_chat_interface(chat_fn: Callable) -> gr.ChatInterface:
    """
    Create reusable chat interface component.

    Args:
        chat_fn: Chat handler function that takes (message, history) and yields responses

    Returns:
        Gradio ChatInterface component configured for negotiation
    """
    return gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        chatbot=gr.Chatbot(height=500, type="messages"),
        textbox=gr.Textbox(
            placeholder="Type your message...",
            container=False,
            scale=7
        ),
        submit_btn="Send",
        concurrency_limit=1  # Ensures turn-based behavior
    )
