"""
Thank you page shown after completing all negotiations.
"""

import os

import gradio as gr
from typing import Tuple


def create_thank_you_page() -> Tuple[gr.Column]:
    """
    Create thank you page shown after all 4 negotiations are complete.

    Returns:
        Tuple containing:
        - thank_you_page: gr.Column containing the thank you UI
    """

    prolific_code = os.environ["PROLIFIC_COMPLETION_CODE"]

    with gr.Column(visible=False) as thank_you_page:
        gr.Markdown("# Thank You! (Teşekkürler!)")
        gr.Markdown("")
        gr.Markdown("## You have completed all negotiations.")
        gr.Markdown("## Tüm müzakereleri tamamladınız.")
        gr.Markdown("")
        gr.Markdown("Your responses have been recorded. You may now close this window.")
        gr.Markdown("Yanıtlarınız kaydedildi. Artık bu pencereyi kapatabilirsiniz.")
        gr.HTML(f"""
        <div style="margin: 32px auto; max-width: 560px; padding: 20px 24px;
                    border: 1px solid #7c3aed !important;
                    background-color: #4c1d95 !important;
                    border-radius: 8px; color: #ffffff !important;">
            <div style="font-weight: 600; margin-bottom: 8px;
                        color: #ffffff !important;">
                If you are a Prolific participant:
            </div>
            <div style="margin-bottom: 12px; color: #ede9fe !important;">
                Please paste the following code into Prolific to confirm completion.
            </div>
            <div style="font-family: 'SFMono-Regular', Menlo, Consolas, monospace;
                        font-size: 1.4rem; font-weight: 700; letter-spacing: 0.08em;
                        text-align: center; padding: 12px 16px;
                        background-color: #ffffff !important;
                        border: 2px solid #c4b5fd !important;
                        border-radius: 6px;
                        color: #4c1d95 !important;
                        user-select: all;">
                {prolific_code}
            </div>
        </div>
        """)

    return (thank_you_page,)
