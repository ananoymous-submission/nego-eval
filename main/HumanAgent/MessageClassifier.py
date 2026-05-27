"""
Message Classification Module

Classifies negotiation messages into types: offer or acceptance.
Uses domain-aware classification based on preference structure.
"""

import dspy
from typing import Dict, List
from main.nenv.MessageType import MessageType
from main.llm_components.base_component import Component
from main.llm_components.preference_utils import load_preference_context

class MessageClassificationSignature(dspy.Signature):
    """
    Classify the type of negotiation message using domain knowledge.

    Message Types:

    - Offer: Contains negotiation offer content (new offer or counter-offer). This does not have to be a complete offer.
        For counter-offers, consider chat history to identify references to previous offers and include agreed values.
        Any message that contains any of the issues/values from the preference structure is considered an offer.
        Messages that don't clearly fit acceptance should be classified as offer.

    - Acceptance: Accepts the current/last offer.
        Whether this becomes full Accept or SubAgreement is decided downstream from the last received bid scope.

    **Critical**: Offers supercede acceptance in classification. If a message includes new issue/value content it is an offer.
    **Critical**: For counter-offers building on previous offers, consider agreed values from chat history.
    """

    message: str = dspy.InputField(desc="Natural language message from human")
    chat_history: List[str] = dspy.InputField(desc="Previous conversation context for counter-offer understanding")
    preference: str = dspy.InputField(desc="Negotiation domain preferences and available issues/values")

    message_type: str = dspy.OutputField(desc="Message type: 'offer' or 'acceptance'")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0 for the classification")


class MessageClassifier(Component):
    """
    Classifies negotiation messages into types: offer or acceptance.

    Uses domain-aware classification based on negotiation preference structure.
    """

    def __init__(self, profile_json_path: str, model_name: str = None, temperature: float = 0.0):
        """
        Initialize the message classifier.

        Args:
            profile_json_path: Path to the profile JSON file
            model_name: Language model to use for classification
            temperature: Sampling temperature for the model
        """
        
        self.preference_context = load_preference_context(profile_json_path)

        super().__init__(
            component_name="MessageClassifier",
            signature=MessageClassificationSignature,
            model_name=model_name,
            temperature=temperature
        )

    def classify(self, message: str, chat_history: List[str] = None) -> Dict[str, any]:
        """
        Classify a negotiation message into its type.

        Args:
            message: Natural language negotiation message
            chat_history: Previous conversation context

        Returns:
            Dictionary containing:
            - 'type': MessageType enum value
            - 'confidence': Classification confidence score (0.0-1.0)
        """
        classification = self.forward(
            message=message,
            chat_history=chat_history or [],
            preference=self.preference_context
        )

        # Convert model output into the 2-way decision used by HumanAgent:
        # OFFER vs ACCEPT. Sub-agreement phrasing is treated as ACCEPT and
        # resolved downstream based on last received bid completeness.
        message_type_str = classification.message_type.lower()
        if message_type_str == 'offer':
            message_type = MessageType.OFFER
        elif message_type_str in ['acceptance', 'accept', 'sub_agreement', 'subagreement']:
            message_type = MessageType.ACCEPT
        else:
            # Unrecognized type — default to offer so CallbackHumanAgent
            # can attempt bid extraction and give a meaningful error.
            message_type = MessageType.OFFER

        return {
            'type': message_type,
            'confidence': classification.confidence
        }
