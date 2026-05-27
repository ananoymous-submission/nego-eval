"""
Preference confirmation page shown after elicitation, before negotiation starts.
Displays the full preference profile (issue weights + value preferences) and lets
the user confirm before the session is initialized.
"""

import gradio as gr
from typing import Tuple


def create_preference_confirmation_page(language: str) -> Tuple[gr.Column, gr.Button, gr.HTML]:
    """Create the preference confirmation page.

    Returns:
        Tuple of (page column, confirm button, preferences html component)
    """
    if language == "Türkçe":
        title = "Müzakere Profiliniz"
        weights_title = "Atanan Konu Öncelikleri"
        weights_note = (
            "Bu çalışmada her konunun önemi size önceden atanmıştır. "
            "Müzakere sırasında bu önceliklere göre kararlar vermenizi bekliyoruz."
        )
        value_note = (
            "Her tercih için seçenekleriz aşşağıda gösterildiği gibidir: "
            "en çok tercih edilenden en az tercih edilene."
        )
        confirm_text = "Anladım, Müzakereye Başla"
    else:
        title = "Your Negotiation Profile"
        weights_title = "Assigned Issue Priorities"
        weights_note = (
            "In this study, the importance of each topic has been pre-assigned to you. "
            "We expect you to make decisions in the negotiation according to these priorities."
        )
        value_note = (
            "For each topic, you ranked the options as below: "
            "most preferred to least preferred."
        )
        confirm_text = "I Understand, Start Negotiation"

    with gr.Column(visible=False) as confirm_page:
        gr.HTML(f"""
        <style>
            .confirm-container {{
                max-width: 900px;
                margin: auto;
                padding: 30px;
                background: #1f2937;
                border-radius: 12px;
            }}
            .confirm-section-title {{
                color: #e5e7eb;
                font-size: 1.1em;
                font-weight: 600;
                margin: 20px 0 8px 0;
                border-bottom: 2px solid #374151;
                padding-bottom: 6px;
            }}
            .confirm-note {{
                background: #374151;
                border-left: 4px solid #60a5fa;
                border-radius: 6px;
                padding: 12px 16px;
                color: #d1d5db;
                font-size: 0.92em;
                margin-bottom: 16px;
            }}
            .confirm-note.green {{
                border-left-color: #34d399;
            }}
        </style>
        """)

        with gr.Column(elem_classes="confirm-container"):
            gr.Markdown(f"# {title}")

            gr.HTML(f"<div class='confirm-section-title'>{weights_title}</div>")
            gr.HTML(f"<div class='confirm-note'>{weights_note}</div>")

            prefs_html = gr.HTML("")

            gr.Markdown("---")
            confirm_btn = gr.Button(confirm_text, variant="primary", size="lg")

    return confirm_page, confirm_btn, prefs_html, value_note
