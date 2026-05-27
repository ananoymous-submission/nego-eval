"""
Frontend pages module.

This module contains individual page components for the Gradio application.
Each page is a self-contained module that returns its UI and event handlers.
"""

from .username import create_username_page
from .survey import create_survey_page
from .negotiation import create_negotiation_page
from .thank_you import create_thank_you_page

__all__ = [
    "create_username_page",
    "create_survey_page",
    "create_negotiation_page",
    "create_thank_you_page",
]
