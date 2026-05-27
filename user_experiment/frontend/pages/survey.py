"""
Reusable survey page for collecting user data via Google Forms.
"""

import gradio as gr
from typing import Tuple, Dict, Any


def create_survey_page(
    survey_url: str,
    title: str = "Pre-Negotiation Survey (Deney Öncesi Anket)",
    subtitle: str = "Please complete the survey below before continuing. / Lütfen müzakereye devam etmeden önce aşağıdaki anketi doldurun.",
    button_text: str = "Continue (Devam Et)"
) -> Tuple[gr.Column, gr.Button, list, list, callable]:
    """
    Create survey page UI (reusable for pre and post surveys).

    Args:
        survey_url: Google Form embed URL
        title: Page title
        subtitle: Instructions text
        button_text: Submit button text

    Returns:
        Tuple of:
        - survey_page: gr.Column containing the survey UI
        - survey_submit_btn: gr.Button for submission
        - survey_inputs: List of input components (empty for Google Form)
        - survey_outputs: List of output components
        - handle_survey_submit: Submit handler function
    """

    with gr.Column(visible=False) as survey_page:
        gr.Markdown(f"## {title}")
        gr.Markdown(subtitle)

        # Hard warning: survey is mid-experiment, NOT the final page.
        gr.Markdown("""
        <div style="background-color: #7f1d1d !important;
                    border: 2px solid #ef4444 !important;
                    padding: 14px 18px; margin: 16px 0;
                    border-radius: 6px;
                    color: #ffffff !important;
                    font-weight: 600;">
            <div style="color: #ffffff !important; margin-bottom: 6px;">
                ⚠️ THIS IS NOT THE FINAL PAGE. PLEASE CONTINUE THE EXPERIMENT AFTER YOU HAVE FILLED OUT THE SURVEY.
            </div>
            <div style="color: #ffffff !important;">
                ⚠️ BU SON SAYFA DEĞİL. ANKETİ DOLDURDUKTAN SONRA LÜTFEN DENEYE DEVAM EDİN.
            </div>
        </div>
        """)

        # Important notice about multi-page survey
        gr.Markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin: 16px 0; border-radius: 4px; color: #856404;">
            <strong style="color: #856404;">⚠️ Important / Önemli:</strong><br/>
            <strong style="color: #856404;">English:</strong> This survey has multiple pages. Please continue until you see the "You have completed the survey" message.<br/>
            <strong style="color: #856404;">Türkçe:</strong> Bu anket birden fazla sayfadan oluşmaktadır. Lütfen "Anketi tamamladınız" mesajını görene kadar devam edin.
        </div>
        """)

        # Embedded Google Form
        gr.HTML(f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <iframe
                    src="{survey_url}"
                    width="640"
                    height="800"
                    frameborder="0"
                    marginheight="0"
                    marginwidth="0"
                    style="border: 1px solid #e0e0e0; border-radius: 8px;">
                    Loading survey...
                </iframe>
            </div>
        """)

        gr.Markdown("---")
        gr.Markdown("**After completing the survey above, click the button below to continue:**\n\n**Yukarıdaki anketi doldurduktan sonra, devam etmek için aşağıdaki düğmeye tıklayın:**")

        survey_error = gr.Markdown("", visible=False)
        survey_submit_btn = gr.Button(
            button_text,
            variant="primary",
            size="lg"
        )

    # Define submit handler
    def handle_survey_submit(state: Dict[str, Any]) -> Tuple[str, gr.update, gr.update, Dict[str, Any]]:
        """
        Handle survey submission (trust-based - user confirms they completed the form).

        Args:
            state: Current user state

        Returns:
            Tuple of (error_message, survey_page_update, next_page_update, updated_state)
        """
        # Track completed surveys
        if "surveys_completed" not in state:
            state["surveys_completed"] = []
        state["surveys_completed"].append(survey_url)

        # Success - hide survey, show next page
        return (
            "",                            # No error
            gr.update(visible=False),      # Hide survey
            gr.update(visible=True),       # Show next page
            state
        )

    # No input components for Google Form (it's embedded)
    survey_inputs = []
    survey_outputs = [survey_error]

    # Return page and components for event wiring
    return (
        survey_page,
        survey_submit_btn,
        survey_inputs,
        survey_outputs,
        handle_survey_submit
    )
