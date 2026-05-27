"""Offer panel HTML generator for the negotiation UI.

Generates a panel below the chat input where users can click to select
issue values. The selected bid is stored in a hidden Gradio textbox and
sent alongside the chat message, bypassing LLM bid extraction.

NOTE: The JS logic lives in app.py's head= script (window.initOfferPanel).
      Scripts inside gr.HTML components aren't executed by browsers when
      injected via innerHTML — so we use an <img onload> trigger instead.
"""

import json


def generate_offer_panel_html(profile_path: str, language: str, page_idx: int) -> str:
    """Generate offer panel HTML for the given domain profile.

    Args:
        profile_path: Path to the user's saved preference profile JSON.
        language: "English" or "Türkçe".
        page_idx: 1 or 2 — used to scope JS to the correct hidden textbox.

    Returns:
        HTML string for the offer panel.
    """
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile = json.load(f)

    issues = profile["issues"]  # {issue_name: {value_name: utility}}
    issue_weights = profile["issueWeights"]  # {issue_name: weight}
    panel_id = f"offer_panel_{page_idx}"
    utility_data_id = f"offer_utility_data_{page_idx}"

    is_turkish = language == "Türkçe"
    panel_title = "Teklifiniz" if is_turkish else "Your Offer"
    utility_title = "Skor" if is_turkish else "Utility"
    utility_meta_label = "konu seçildi" if is_turkish else "issues selected"
    user_offer_utility_label = "Skor" if is_turkish else "Utility"

    # Build issue rows
    issues_html = ""
    for issue_name, values in issues.items():
        safe_issue = issue_name.replace("'", "\\'").replace('"', '\\"')
        value_btns = ""
        for value_name in values.keys():
            safe_value = value_name.replace("'", "\\'").replace('"', '\\"')
            value_btns += (
                f'<button class="offer-val-btn" '
                f'onclick="offerSelectValue_{page_idx}(\'{safe_issue}\', \'{safe_value}\', this)">'
                f'{value_name}</button>'
            )
        issues_html += f"""
        <div class="offer-issue-row">
            <div class="offer-issue-label">{issue_name}</div>
            <div class="offer-value-btns">{value_btns}</div>
        </div>"""

    # Tiny transparent GIF — executes onload to init JS (scripts in innerHTML don't run)
    BLANK_GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    utility_data_json = json.dumps({
        "issueWeights": issue_weights,
        "issues": issues,
    }).replace("</", "<\\/")

    return f"""
<style>
#{panel_id} {{
    background: #1f2937;
    border-top: 2px solid #374151;
    padding: 12px 16px 10px;
    border-radius: 0 0 8px 8px;
}}
#{panel_id} .offer-panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}}
#{panel_id} .offer-panel-title {{
    color: #e5e7eb;
    font-weight: 600;
    font-size: 0.95em;
}}
#{panel_id} .offer-utility-wrap {{
    margin-bottom: 12px;
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 10px;
}}
#{panel_id} .offer-utility-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}}
#{panel_id} .offer-utility-title {{
    color: #d1d5db;
    font-size: 0.8em;
    font-weight: 600;
}}
#{panel_id} .offer-utility-value {{
    color: #6ee7b7;
    font-size: 0.86em;
    font-weight: 700;
}}
#{panel_id} .offer-utility-track {{
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: #374151;
    overflow: hidden;
}}
#{panel_id} .offer-utility-fill {{
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, #059669 0%, #10b981 100%);
    transition: width 0.16s ease-out;
}}
#{panel_id} .offer-utility-meta {{
    margin-top: 6px;
    color: #9ca3af;
    font-size: 0.75em;
}}
#{panel_id} .offer-issue-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
    flex-wrap: wrap;
}}
#{panel_id} .offer-issue-label {{
    min-width: 115px;
    color: #d1d5db;
    font-size: 0.84em;
    font-weight: 500;
    text-align: right;
}}
#{panel_id} .offer-value-btns {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}}
#{panel_id} .offer-val-btn {{
    background: #374151;
    color: #d1d5db;
    border: 1.5px solid #4b5563;
    border-radius: 6px;
    padding: 4px 10px;
    cursor: pointer;
    font-size: 0.81em;
    transition: all 0.12s;
}}
#{panel_id} .offer-val-btn:hover {{
    border-color: #6b7280;
    background: #4b5563;
}}
#{panel_id} .offer-val-btn.selected {{
    background: #1d4ed8;
    border-color: #2563eb;
    color: #ffffff;
    font-weight: 600;
}}
#{panel_id} .offer-val-btn.disabled {{
    opacity: 0.45;
    cursor: not-allowed;
}}
#{panel_id} .offer-val-btn.locked {{
    background: #059669;
    border-color: #10b981;
    color: #d1fae5;
    opacity: 1;
}}
#{panel_id} .offer-issue-row.locked .offer-issue-label {{
    color: #6ee7b7;
    font-weight: 600;
}}
</style>

<div id="{panel_id}" data-page-idx="{page_idx}" data-user-utility-label="{user_offer_utility_label}">
    <div class="offer-panel-header">
        <span class="offer-panel-title">{panel_title}</span>
    </div>
    <div class="offer-utility-wrap" data-utility-meta-label="{utility_meta_label}">
        <div class="offer-utility-head">
            <span class="offer-utility-title">{utility_title}</span>
            <span class="offer-utility-value">0.0%</span>
        </div>
        <div class="offer-utility-track">
            <div class="offer-utility-fill"></div>
        </div>
        <div class="offer-utility-meta">0/{len(issues)} {utility_meta_label}</div>
    </div>
    {issues_html}
</div>
<script id="{utility_data_id}" type="application/json">{utility_data_json}</script>

<img src="{BLANK_GIF}"
     onload="if(window.initOfferPanel)window.initOfferPanel({page_idx})"
     style="display:none">
"""
