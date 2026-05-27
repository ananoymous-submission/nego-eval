import json
from typing import Dict, Any
from functools import lru_cache

@lru_cache(maxsize=32)
def load_preference_context(profile_json_path: str) -> str:
    """
    Load and cache preference context from JSON file.
    
    Args:
        profile_json_path: Path to the preference profile JSON file
        
    Returns:
        Formatted preference context string for DSPy signatures
    """
    with open(profile_json_path, 'r', encoding='utf-8') as f:
        preference_profile = json.load(f)
    
    return create_preference_context(preference_profile)


def create_preference_context(preference_profile: Dict[str, Any]) -> str:
    """
    Create a systematic preference context string for DSPy signature injection.
    
    Args:
        preference_profile: Agent preferences from profile.json
        
    Returns:
        Formatted preference context string
    """
    issues = preference_profile["issues"]
    weights = preference_profile["issueWeights"]
    
    # Create preference context following 2025 context engineering best practices
    context_parts = [
        "CRITICAL: Do not consider any offers that include issues or values that are not in the domain.",
        "CRITICAL: If you do receive an offer that includes issues or values that are not in the domain, try to remind the valid issues and values to the other party.\n",
        "Your preferences are:",
    ]
    
    # Add issue importance (weights)
    context_parts.append("Issue importance rankings (higher weight = more important):")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for issue, weight in sorted_weights:
        context_parts.append(f"- {issue}: {weight:.1f} importance")
    
    # Add value preferences for each issue
    context_parts.append("\nYour value preferences for each issue:")
    for issue, values in issues.items():
        weight = weights[issue]
        context_parts.append(f"\n{issue} (importance: {weight:.1f}):")
        
        # Sort values by preference (higher utility = more preferred)
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        for value, utility in sorted_values:
            context_parts.append(f"  - {value}: {utility:.2f} utility")
    
    return "\n".join(context_parts)
