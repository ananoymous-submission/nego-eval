"""Preferences display component for negotiation UI."""

import json
import gradio as gr


def create_bar_html(label: str, value: float, max_width: int = 200) -> str:
    """
    Create HTML for a single horizontal bar.

    Args:
        label: Bar label text
        value: Normalized value (0.0 to 1.0)
        max_width: Maximum bar width in pixels

    Returns:
        HTML string for the bar
    """
    percentage = int(value * 100)
    bar_width = int(value * max_width)

    # Color: brighter green for higher values (for dark background)
    # Use a gradient from medium green to bright green
    green_intensity = int(150 + (105 * value))  # 150-255 range for better visibility on dark
    color = f"rgb(34, {green_intensity}, 80)"

    return f"""
    <div style="margin: 6px 0; display: flex; align-items: center; gap: 10px;">
        <div style="min-width: 120px; text-align: right; font-size: 0.9em; color: #e5e7eb;">
            {label}
        </div>
        <div style="flex: 1; background: #374151; border-radius: 4px; height: 24px; position: relative; max-width: {max_width}px;">
            <div style="background: {color}; height: 100%; border-radius: 4px; width: {bar_width}px; transition: width 0.3s ease;"></div>
        </div>
        <div style="min-width: 45px; font-size: 0.9em; font-weight: 600; color: {color};">
            {percentage}%
        </div>
    </div>
    """


def generate_preferences_html(profile_path: str, domain: str, value_note: str = None) -> str:
    """
    Generate HTML content for preferences display.

    Args:
        profile_path: Path to preference profile JSON file
        domain: Domain type ("holiday" or "resource")

    Returns:
        HTML string with bar graph preferences

    Raises:
        FileNotFoundError: If profile path doesn't exist
        json.JSONDecodeError: If profile is invalid JSON
    """
    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    html_parts = []

    # Issue Importance section
    html_parts.append("""
        <div style="margin-bottom: 25px;">
            <h4 style="margin: 0 0 12px 0; color: #e5e7eb; font-size: 1em; font-weight: 600;">
                Issue Importance ( Konu Önemleri )
            </h4>
    """)

    issue_weights = data['issueWeights']
    for issue_name, weight in sorted(issue_weights.items(), key=lambda x: x[1], reverse=True):
        html_parts.append(create_bar_html(issue_name, weight, max_width=250))

    html_parts.append("</div>")

    # Value preferences per issue (collapsible sections)
    # Only display for holiday domain - resource domain uses fixed values
    if domain == "holiday":
        note_html = ""
        if value_note:
            note_html = f"""<div style="background:#374151;border-left:4px solid #34d399;border-radius:6px;padding:10px 14px;color:#d1d5db;font-size:0.88em;margin-bottom:14px;">{value_note}</div>"""
        html_parts.append(f"""
            <div style="margin-top: 20px;">
                <h4 style="margin: 0 0 12px 0; color: #e5e7eb; font-size: 1em; font-weight: 600;">
                    Value Preferences ( Değer Önemleri )
                </h4>
                {note_html}
        """)

        value_weights = data['issues']

        for idx, (issue_name, values) in enumerate(value_weights.items()):
            # Create collapsible section for each issue (open by default)
            section_id = f"issue_{idx}"

            html_parts.append(f"""
            <details open style="margin: 10px 0; background: #1f2937; border-radius: 6px; padding: 8px 12px; border: 1px solid #374151;">
                <summary style="cursor: pointer; font-weight: 600; color: #f3f4f6; user-select: none; list-style: none;">
                    <span class="arrow"></span>
                    {issue_name}
                </summary>
                <div style="margin-top: 12px; padding-left: 20px;">
            """)

            # Add bars for each value
            for value, weight in sorted(values.items(), key=lambda x: x[1], reverse=True):
                html_parts.append(create_bar_html(value, weight, max_width=180))

            html_parts.append("""
                </div>
            </details>
            """)

        html_parts.append("</div>")

    html_parts.append("</div>")

    # Add CSS to style details/summary
    html_parts.append("""
    <style>
        details > summary {
            list-style: none;
        }
        details > summary::-webkit-details-marker {
            display: none;
        }
        details[open] > summary .arrow::before {
            content: "▼";
        }
        details:not([open]) > summary .arrow::before {
            content: "▶";
        }
        details > summary .arrow {
            display: inline-block;
            margin-right: 8px;
            width: 12px;
        }
        details > summary:hover {
            background: #374151;
            border-radius: 4px;
        }
    </style>
    """)

    return "".join(html_parts)


def create_preferences_display() -> gr.HTML:
    """
    Create an empty preferences display component.

    Returns:
        Gradio HTML component (initially empty, updated dynamically)
    """
    return gr.HTML("")
