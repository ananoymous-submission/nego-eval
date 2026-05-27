import json
from typing import Dict, Any


def load_preference_context(profile_json_path: str) -> str:
    """Read a preference profile from disk and format it for the LLM prompt.

    Not cached — user-elicited profiles can be overwritten between sessions
    at the same path, and a cached version would silently misrepresent the
    weights to the LLM.
    """
    with open(profile_json_path, 'r', encoding='utf-8') as f:
        preference_profile = json.load(f)

    return create_preference_context(preference_profile)


def create_preference_context(preference_profile: Dict[str, Any]) -> str:
    issues = preference_profile["issues"]
    weights = preference_profile["issueWeights"]

    context_parts = [
        "CRITICAL: Do not consider any offers that include issues or values that are not in the domain.",
        "CRITICAL: If you do receive an offer that includes issues or values that are not in the domain, try to remind the valid issues and values to the other party.\n",
        "Your preferences are:",
    ]

    # Use enough precision that weights like 0.15, 0.25 don't collapse onto
    # the same digit — the LLM was being shown 0.1 for 0.15 with .1f.
    context_parts.append("Issue importance rankings (higher weight = more important):")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for issue, weight in sorted_weights:
        context_parts.append(f"- {issue}: {weight:.3f} importance")

    context_parts.append("\nYour value preferences for each issue:")
    for issue, values in issues.items():
        weight = weights[issue]
        context_parts.append(f"\n{issue} (importance: {weight:.3f}):")

        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        for value, utility in sorted_values:
            context_parts.append(f"  - {value}: {utility:.3f} utility")

    return "\n".join(context_parts)
