"""
Preference elicitation page with horizontal drag-and-drop ordering.
Users rank their preferred values for each issue (issue weights are fixed).
Visual: [Least Preferred] < [Item] < [Most Preferred]
"""

import gradio as gr
import json
from typing import Tuple, Dict, List


def create_preference_elicitation_page(
    domain: str,
    language: str
) -> Tuple[gr.Column, gr.Button, List[gr.Slider], Dict[str, List[gr.Slider]], gr.Markdown, gr.Markdown, List[str], Dict[str, List[str]], gr.Textbox]:
    """Create preference elicitation UI with horizontal drag-and-drop ordering.
    
    Returns:
        Tuple containing: page, button, issue_sliders, value_sliders_dict, status_msg, 
        sum_display, issue_names, value_names_per_issue, order_textbox
    """
    
    # Load profile data
    language_dir = "turkisch" if language == "Türkçe" else "englisch"
    profile_path = f"main/domains/{language_dir}/{domain}/profile.json"

    with open(profile_path, 'r', encoding='utf-8') as f:
        template_profile = json.load(f)

    issue_names = list(template_profile["issueWeights"].keys())
    value_names_per_issue = {
        issue: list(template_profile["issues"][issue].keys()) 
        for issue in issue_names
    }

    # Text based on domain/language
    if domain == "holiday":
        session_title = "Holiday Session (Tatil Seansı)"
    else:
        session_title = "Island Session (Issız Ada Seansı)"

    if language == "Türkçe":
        subtitle = "Her seçenek için tercihlerinizi sürükleyerek sıralayın"
        values_title = "Seçenek Tercihleri"
        button_text = "Müzakereye Başla"
    else:
        subtitle = "Drag to rank your preferences for each option"
        values_title = "Option Preferences"
        button_text = "Start Negotiation"

    # Use FIXED IDs
    issue_container_id = f"issue_sort_{domain}_{language[:2]}"
    order_textbox_id = f"order_data_{domain}_{language[:2]}"
    
    with gr.Column(visible=False) as pref_page:
        # CSS
        gr.HTML(f"""
        <style>
            .pref-container {{
                max-width: 1300px;
                margin: auto;
                padding: 30px;
                background: #1f2937;
                border-radius: 12px;
            }}
            .section-title {{
                color: #e5e7eb;
                font-size: 1.2em;
                font-weight: 600;
                margin: 25px 0 10px 0;
                border-bottom: 2px solid #374151;
                padding-bottom: 8px;
            }}
            .section-desc {{
                color: #9ca3af;
                font-size: 0.95em;
                margin-bottom: 15px;
            }}
            .sortable-row {{
                display: flex;
                align-items: center;
                gap: 30px;
                padding: 15px;
                background: #111827;
                border-radius: 8px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }}
            .sortable-item {{
                background: linear-gradient(135deg, #4f46e5, #7c3aed);
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: grab;
                font-weight: 500;
                user-select: none;
                border: 2px solid transparent;
                position: relative;
            }}
            .sortable-item:not(:last-child)::after {{
                content: '<';
                position: absolute;
                right: -24px;
                top: 50%;
                transform: translateY(-50%);
                color: #9ca3af;
                font-size: 1.4em;
                font-weight: bold;
                pointer-events: none;
            }}
            .sortable-item:hover {{
                border-color: #a78bfa;
            }}
            .sortable-item:active {{
                cursor: grabbing;
            }}
            .sortable-ghost {{
                opacity: 0.4;
                background: #22c55e !important;
            }}
            .sortable-chosen {{
                border-color: #22c55e !important;
            }}
            .scenario-box {{
                background-color: #374151;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid;
            }}
            .scenario-box p {{ color: #e5e7eb; margin: 0; }}
            .explanation-box {{
                background-color: #374151;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #60a5fa;
            }}
            .explanation-box p {{ color: #e5e7eb; margin: 0 0 10px 0; }}
            .explanation-box p:last-child {{ margin: 0; }}
        </style>
        """)

        with gr.Column(elem_classes="pref-container"):
            gr.Markdown(f"# {session_title}")
            gr.Markdown(f"### {subtitle}")
            status_msg = gr.Markdown("", visible=True)

            # SCENARIO
            issues_list = ", ".join(issue_names)
            if domain == "holiday":
                border_color = "#8b5cf6"
                scenario_html = f"""<div class="scenario-box" style="border-left-color: {border_color};"><p><strong>📋 {'Scenario' if language=='English' else 'Senaryo'}:</strong> {'You are planning a holiday with a friend. You need to negotiate and reach an agreement on: ' if language=='English' else 'Bir arkadaşınızla tatil planlıyorsunuz. Aşağıdaki konularda anlaşmanız gerekiyor: '}<strong>{issues_list}</strong></p></div>"""
            else:
                border_color = "#10b981"
                scenario_html = f"""<div class="scenario-box" style="border-left-color: {border_color};"><p><strong>🏝️ {'Scenario' if language=='English' else 'Senaryo'}:</strong> {'You are stranded on an island with a friend. You need to negotiate how to split: ' if language=='English' else 'Bir arkadaşınızla bir adada mahsur kaldınız. Paylaşmanız gereken: '}<strong>{issues_list}</strong></p></div>"""
            gr.HTML(scenario_html)

            # Hidden sliders - Set initial weights based on initial order
            issue_sliders = []
            num_issues = len(issue_names)
            reversed_initial = list(reversed(issue_names))
            initial_raw = {name: num_issues - i for i, name in enumerate(reversed_initial)}
            initial_total = sum(initial_raw.values())
            initial_weights = {k: (v/initial_total)*100 for k, v in initial_raw.items()}
            
            for name in issue_names:
                s = gr.Slider(minimum=0, maximum=100, value=initial_weights[name], visible=False, label=name)
                issue_sliders.append(s)

            # HIDDEN TEXTBOX to store current order - JS will update this on every drag
            order_textbox = gr.Textbox(
                value=json.dumps({"issues": issue_names, "values": value_names_per_issue}),
                visible=False,
                elem_id=order_textbox_id
            )

            # VALUES (Holiday only)
            value_sliders_dict = {}
            
            if domain == "holiday":
                gr.HTML(f"<div class='section-title'>{values_title}</div>")
                val_explanation = f"""<div class="explanation-box" style="border-left-color: #34d399;"><p><strong>{'Drag boxes to rank your options. Left = least preferred &nbsp;&lt;&nbsp; Right = most preferred.' if language=='English' else 'Kutuları sürükleyin. Sol = en az tercih edilen &nbsp;&lt;&nbsp; Sağ = en çok tercih edilen.'}</strong></p></div>"""
                gr.HTML(val_explanation)

                for issue_name in issue_names:
                    with gr.Accordion(issue_name, open=True):
                        values = value_names_per_issue[issue_name]

                        rev_vals = list(reversed(values))
                        num_vals = len(rev_vals)
                        init_vals = {name: (num_vals - i) * (100 / num_vals) for i, name in enumerate(rev_vals)}
                        
                        v_sliders = []
                        value_sliders_dict[issue_name] = []
                        for v in values:
                            s = gr.Slider(minimum=0, maximum=100, value=init_vals[v], visible=False, label=v)
                            v_sliders.append(s)
                            value_sliders_dict[issue_name].append(s)

                        v_container_id = f"values_{issue_name.replace(' ', '_')}_{domain}_{language[:2]}"
                        
                        v_items_html = ""
                        for v in values:
                            v_items_html += f'<div class="sortable-item" data-id="{v}">{v}</div>'

                        gr.HTML(f"""<div id="{v_container_id}" class="sortable-row">{v_items_html}</div>""")
                        gr.HTML(value=f"<img src='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7' onload=\"if(window.initDragDrop)window.initDragDrop('{v_container_id}','{order_textbox_id}')\" style='display:none'>")

            sum_display = gr.Markdown("", visible=False)
            gr.Markdown("---")
            submit_btn = gr.Button(button_text, variant="primary", size="lg")

    return pref_page, submit_btn, issue_sliders, value_sliders_dict, status_msg, sum_display, issue_names, value_names_per_issue, order_textbox
