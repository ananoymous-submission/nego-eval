from typing import List, Optional
from main.nenv.Agent import AbstractAgent
from main.nenv.Preference import Preference
from main.nenv.Action import Action
from main.nenv.Bid import Bid
from main.HumanAgent.MessageClassifier import MessageClassifier


class HumanAgent(AbstractAgent):

    def __init__(self, preference: Preference, session_time: int, estimators: List,
                 name: str = "Human", model_name: Optional[str] = None):
        """ALTERNATING-only human agent. Free-text input goes through
        MessageClassifier to decide offer-vs-accept; structured bids come
        from the UI panel."""
        super().__init__(preference, session_time, estimators)
        self._name = name
        self.classifier = MessageClassifier(preference.profile_json_path, model_name=model_name)
        
    @property
    def name(self) -> str:
        return self._name
    
    def receive_offer(self, bid: Bid, t: float):
        super().receive_offer(bid, t)

    def receive_action(self, action, t: float, chat_history: List[str] = None):
        super().receive_action(action, t, chat_history)

    def _format_domain_info(self) -> str:
        """Format domain information for human display."""
        lines = []
        for issue in self.preference.issues:
            values = ', '.join(issue.values)
            lines.append(f"  {issue.name}: {values}")
        return "Negotiation Issues:\n" + '\n'.join(lines)
    
    def _format_preference_info(self) -> str:
        """Format preference information for human display."""
        lines = ["Issue importance (higher = more important):"]
        
        # Sort by weight descending
        sorted_weights = sorted(
            self.preference.issue_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Display issue weights
        for issue, weight in sorted_weights:
            lines.append(f"  {issue.name}: {weight:.2f}")
            
        lines.append("\nValue preferences per issue:")
        
        # Display value weights for each issue
        for issue in self.preference.issues:
            lines.append(f"\n  {issue.name}:")
            value_weights = self.preference.value_weights[issue]
            # Sort values by weight descending
            sorted_values = sorted(
                value_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for value, weight in sorted_values:
                lines.append(f"    {value}: {weight:.2f}")
            
        return '\n'.join(lines)
    
    def _format_bid_for_human(self, bid: Bid) -> str:
        """Format a bid in human-readable format."""
        if not bid or not bid.content:
            return "No specific offer"
            
        parts = []
        for issue, value in bid.content.items():
            parts.append(f"{issue.name}: {value}")
        
        return ", ".join(parts)
