"""Multi-user Gradio application for negotiation chatbot.

Supports username entry → survey → preference elicitation → negotiation pipeline with isolated per-user sessions.
"""

import os
import time
import queue
import json
import base64
import math
import threading
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gradio as gr

# Load env before importing services/components that may read tracing config at import time.
load_dotenv()

from user_experiment.backend.service.negotiation_service import NegotiationService
from user_experiment.backend.config.settings import AppConfig
from user_experiment.backend.database.db_service import DatabaseService
from user_experiment.frontend.pages import create_username_page, create_survey_page, create_negotiation_page, create_thank_you_page
from user_experiment.frontend.pages.preference_elicitation import create_preference_elicitation_page
from user_experiment.frontend.pages.preference_confirmation import create_preference_confirmation_page
from user_experiment.frontend.pages.video_explanation import (
    VIDEO_EXPLANATION_CONTINUE_BUTTON_ID,
    create_video_explanation_page,
    build_video_explanation_embed,
)
from user_experiment.frontend.components.timer import update_timer_html
from user_experiment.frontend.components.preferences_display import generate_preferences_html
from user_experiment.frontend.components.offer_panel import generate_offer_panel_html

# STEP 1: Ensure persistent storage directories exist FIRST
from user_experiment.backend.storage.persistent_storage import ensure_storage_dirs
ensure_storage_dirs()

# STEP 2: Initialize configuration, service, and database (depends on data existing)
config = AppConfig()
negotiation_service = NegotiationService(config.session)
db_service = DatabaseService(
    os.getenv("DATABASE_PATH", "negotiation.db")
)

# Set LangSmith project
os.environ["LANGSMITH_PROJECT"] = config.langsmith_project

# Get survey URLs from environment
PRE_SURVEY_LINK = os.getenv("PRE_SURVEY_LINK")
LLM_AGENT_SURVEY_LINK = os.getenv("LLM_AGENT_SURVEY_LINK")
HEURISTIC_AGENT_SURVEY_LINK = os.getenv("HEURISTIC_AGENT_SURVEY_LINK")

# Get video explanation URLs from environment
VIDEO_EXPLANATION_PATH_EN = os.getenv("VIDEO_EXPLANATION_PATH_EN")
VIDEO_EXPLANATION_PATH_TR = os.getenv("VIDEO_EXPLANATION_PATH_TR")

# Validate all required environment variables (no fallbacks!)
if not PRE_SURVEY_LINK:
    raise ValueError("PRE_SURVEY_LINK environment variable is required")
if not LLM_AGENT_SURVEY_LINK:
    raise ValueError("LLM_AGENT_SURVEY_LINK environment variable is required")
if not HEURISTIC_AGENT_SURVEY_LINK:
    raise ValueError("HEURISTIC_AGENT_SURVEY_LINK environment variable is required")
if not VIDEO_EXPLANATION_PATH_EN:
    raise ValueError("VIDEO_EXPLANATION_PATH_EN environment variable is required and must be a YouTube URL")
if not VIDEO_EXPLANATION_PATH_TR:
    raise ValueError("VIDEO_EXPLANATION_PATH_TR environment variable is required and must be a YouTube URL")

# Map internal agent_type → per-session survey URL.
# Internal agent_type "Traditional" maps to the heuristic-condition survey.
AGENT_SURVEY_URLS = {
    "LLM": LLM_AGENT_SURVEY_LINK,
    "Traditional": HEURISTIC_AGENT_SURVEY_LINK,
}

# Round-robin counter so half the users start with the LLM opponent and half
# with the Traditional one. Each user always sees both, in alternating order.
_agent_type_counter = 0
_agent_type_counter_lock = threading.Lock()

# Store queues outside of gr.State (keyed by session_id)
# gr.State cannot contain Queue objects (not deepcopyable)
user_queues = {}


def generate_negotiation_order():
    """Two holiday negotiations per user, one against each AI opponent.

    Returns:
        List of 2 (domain, agent_type) tuples — agent_type ∈ {"LLM", "Traditional"}.
    """
    global _agent_type_counter
    agent_types = ["LLM", "Traditional"]

    with _agent_type_counter_lock:
        first_idx = _agent_type_counter % 2
        _agent_type_counter += 1

    return [
        ("holiday", agent_types[first_idx]),
        ("holiday", agent_types[1 - first_idx]),
    ]


def initialize_user_state():
    """
    Initialize per-user state dictionary.

    Returns:
        State dict with user-specific data (queues created lazily)
    """
    return {
        "user_id": None,
        "username": None,
        "language": None,
        "survey": {},
        "negotiations": [],  # List of 2 (domain, agent_type) tuples
        "current_negotiation_index": 0,  # 0-1
        "session_id": None,
        "session_started": False,
        "negotiation_start_time": None,  # timestamp when user sends first message

        # User preference profiles
        "user_profiles": {},  # {domain: profile_path} - paths to user-created profile.json files
        "llm_profiles": {}    # {domain: profile_path} - paths to inverted profileB.json files
    }



def normalize_issue_weights_on_change(
    changed_index: int,
    new_value: float,
    all_values: list
) -> list:
    """
    Normalize issue weights when one changes.

    Maintains relative proportions of other issues while ensuring sum = 100.

    Args:
        changed_index: Index of the slider that changed
        new_value: New value for changed slider (0-100)
        all_values: Current values of all sliders

    Returns:
        List of normalized values
    """
    # Ensure new_value is within bounds
    new_value = max(0, min(100, new_value))

    # Calculate sum of others
    other_sum = sum(v for i, v in enumerate(all_values) if i != changed_index)

    # Calculate remaining to distribute
    remaining = 100.0 - new_value

    # Build new values
    new_values = []
    for i, old_value in enumerate(all_values):
        if i == changed_index:
            new_values.append(new_value)
        elif other_sum > 0:
            # Proportionally redistribute
            proportion = old_value / other_sum
            new_values.append(remaining * proportion)
        else:
            # Distribute equally if all others were 0
            num_others = len(all_values) - 1
            new_values.append(remaining / num_others if num_others > 0 else 0)

    return new_values


def update_sum_display(values: list) -> str:
    """
    Update the sum display markdown.

    Args:
        values: List of current slider values

    Returns:
        Markdown string with sum and check mark
    """
    total = sum(values)
    check = "✓" if abs(total - 100) < 0.1 else "✗"
    return f"**Total: {total:.1f}% {check}**"


def get_negotiation_instructions(language: str, domain: str) -> str:
    """Bilingual negotiation rules. The opponent agent type is NOT shown to
    the user — it's the experiment's independent variable."""
    if domain == "holiday":
        domain_desc_en = "You are planning a holiday with a friend. You need to negotiate and reach an agreement on: Accommodation, Destination, Season, Activities, and Transportation."
        domain_desc_tr = "Bir arkadaşınızla tatil planlıyorsunuz. Şu konular üzerinde müzakere edip anlaşmaya varmanız gerekiyor: Konaklama, Destinasyon, Sezon, Aktivite ve Ulaşım."
        examples_en = """
        <li style="color: black;">"Let's go to Paris, stay in a Hostel, in the Summer, go to Museums and use the Bus."</li>
        <li style="color: black;">"How about: Berlin, Hotel, Winter, for Shopping and we take the Subway?"</li>
        <li style="color: black;">"I suggest: Roma, Caravan, Autumn, Museums and rent a Car."</li>
        """
        examples_tr = """
        <li style="color: black;">"Paris'e gidelim, Hostel'de kalalım, Yazın, Müze gezelim ve Otobüs ile gezelim."</li>
        <li style="color: black;">"Şöyle olsa: Berlin, Hotel, Kış, Alışveriş yapalım ve Metro kullanalım?"</li>
        <li style="color: black;">"Önerim: Roma, Karavan, Sonbahar, Müze gezelim ve Araba kiralayalım."</li>
        """

    if language == "Türkçe":
        title = "NASIL MÜZAKERE EDİLİR"
        important_label = "⚠️ ÖNEMLİ"
        rules_desc = "Her teklif tüm konular için bir değer içermelidir."
        what_must_do = "✅ Yapmanız GEREKENLER:"
        what_can_send = "Gönderebileceğiniz mesajlar:"
        scenario_label = "📋 Senaryo:"
        examples_label = "💬 Örnek Teklifler:"
        ready_label = "🚀 Başlamaya Hazır mısınız?"
        ready_text = "Müzakereye başlamak için aşağıdaki sohbete ilk mesajınızı yazın. 15 dakikalık süre, ilk mesajınızı gönderdikten sonra başlayacaktır."
        domain_desc = domain_desc_tr
        examples = examples_tr
        must_do_items = """
        <li style="color: black;">Her teklifinizde tüm konuları belirtin</li>
        """
        can_send_items = """
        <li style="color: black;">Tüm müzakere konuları için bir değer içeren teklifler</li>
        <li style="color: black;">Rakibin teklifini <strong style="color: black;">kabul etme</strong> (örn: "Kabul ediyorum", "Anlaştık", "Tamam")</li>
        """
    else:  # English
        title = "HOW TO NEGOTIATE"
        important_label = "⚠️ IMPORTANT"
        rules_desc = "Every offer must include a value for every issue."
        what_must_do = "✅ What you MUST do:"
        what_can_send = "What you can send:"
        scenario_label = "📋 Scenario:"
        examples_label = "💬 Example Offers:"
        ready_label = "🚀 Ready to Begin?"
        ready_text = "To start the negotiation, type your first message in the chat below. The 15-minute timer will begin after you send your first message."
        domain_desc = domain_desc_en
        examples = examples_en
        must_do_items = """
        <li style="color: black;">Include every issue in every offer you make</li>
        """
        can_send_items = """
        <li style="color: black;">Offers that specify a value for every negotiation issue</li>
        <li style="color: black;"><strong style="color: black;">Acceptance</strong> of opponent's offer (e.g., "I accept", "Deal", "Agreed")</li>
        """

    return f"""
<h1 style="text-align: center; font-size: 40px; font-weight: bold; color: white; margin-bottom: 20px;">
    {title}
</h1>

<div style="background-color: #a8d8ea; border-left: 4px solid #4a90a4; padding: 20px; margin: 15px 0; border-radius: 8px; color: black;">
    <p style="margin: 0 0 15px 0; color: black;"><strong style="color: black;">{important_label}</strong></p>
    <p style="margin: 0 0 10px 0; color: black;">{rules_desc}</p>

    <p style="margin: 15px 0 5px 0; color: black;"><strong style="color: black;">{what_must_do}</strong></p>
    <ul style="margin: 5px 0 10px 20px; color: black;">
        {must_do_items}
    </ul>

    <p style="margin: 15px 0 5px 0; color: black;"><strong style="color: black;">{what_can_send}</strong></p>
    <ol style="margin: 5px 0 0 20px; color: black;">
        {can_send_items}
    </ol>
</div>

<div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 15px 0; border-radius: 8px; color: black;">
    <p style="margin: 0 0 10px 0; color: black;"><strong style="color: black;">{scenario_label}</strong></p>
    <p style="margin: 0 0 20px 0; color: black;">{domain_desc}</p>

    <p style="margin: 0 0 10px 0; color: black;"><strong style="color: black;">{examples_label}</strong></p>
    <ul style="margin: 5px 0 0 20px; color: black;">
        {examples}
    </ul>
</div>

<div style="background-color: #b8e6b8; border-left: 4px solid #4a9d4a; padding: 20px; margin: 15px 0; border-radius: 8px; color: black;">
    <p style="margin: 0 0 8px 0; color: black;"><strong style="color: black;">{ready_label}</strong></p>
    <p style="margin: 0; color: black;">{ready_text}</p>
</div>
"""


def get_human_input_callback(session_id: str):
    """
    Blocking callback for CallbackHumanAgent.

    Args:
        session_id: Session identifier to get queue for

    Returns:
        User's message from input queue
    """
    if session_id not in user_queues:
        raise ValueError(f"No queues found for session {session_id}")
    return user_queues[session_id]["input"].get()


def on_llm_message_callback(session_id: str, message: str):
    """
    Callback when LLM sends a message.

    Args:
        session_id: Session identifier to get queue for
        message: LLM's message
    """
    if session_id not in user_queues:
        raise ValueError(f"No queues found for session {session_id}")
    user_queues[session_id]["output"].put(message)


def _extract_locked_agreements(context) -> dict:
    """Collect locked issues from all recorded sub-agreements in this session."""
    locked = {}
    if not context:
        return locked

    for action in getattr(context.session, "action_history", []) or []:
        if action.__class__.__name__ != "SubAgreement":
            continue

        bid = getattr(action, "bid", None)
        if bid is None:
            continue

        for issue, value in bid.content.items():
            issue_name = issue.name if hasattr(issue, "name") else str(issue)
            locked[issue_name] = value

    return locked


def _get_locked_agreements_json_for_offer_panel(session_id: str) -> str:
    """Return locked agreements as JSON for offer panel preselection."""
    try:
        context = negotiation_service.session_manager.sessions.get(session_id)
        locked = _extract_locked_agreements(context)
        return json.dumps(locked, ensure_ascii=False)
    except Exception as e:
        print(f"[{session_id}] Locked agreements lookup failed: {e}")
        return "{}"


def _get_latest_opponent_display_data(session_id: str, opponent_message: str) -> dict:
    """
    Build metadata for the latest opponent message:
    utility + currently discussed scope + locked issues.
    """
    try:
        context = negotiation_service.session_manager.sessions.get(session_id)
        if not context or not context.session.action_history:
            return {"utility": None, "scope_issues": [], "locked_issues": []}

        locked_map = _extract_locked_agreements(context)
        locked_issues = sorted(locked_map.keys())

        last_action = context.session.action_history[-1]
        last_message = (getattr(last_action, "message", "") or "").strip()
        current_message = (opponent_message or "").strip()

        # Guard against stale actions (e.g., human-side validation errors pushed to output queue).
        if not last_message or not current_message or last_message != current_message:
            return {"utility": None, "scope_issues": [], "locked_issues": locked_issues}

        bid = getattr(last_action, "bid", None)
        if bid is None:
            return {"utility": None, "scope_issues": [], "locked_issues": locked_issues}

        scope_issues = sorted(
            issue.name if hasattr(issue, "name") else str(issue)
            for issue in bid.content.keys()
        )
        active_scope = [name for name in scope_issues if name not in locked_map]

        return {
            "utility": context.session.agentA.preference.get_utility(bid),
            "scope_issues": active_scope,
            "locked_issues": locked_issues,
        }
    except Exception as e:
        print(f"[{session_id}] Utility display lookup failed: {e}")
        return {"utility": None, "scope_issues": [], "locked_issues": []}


def _get_latest_human_display_data(session_id: str, user_message: str) -> dict:
    """Get utility of the latest logged human action that matches current user message."""
    try:
        context = negotiation_service.session_manager.sessions.get(session_id)
        if not context:
            return {"utility": None}

        rows = getattr(getattr(context.session, "session_log", None), "log_rows", {}).get("Session", [])
        if not rows:
            return {"utility": None}

        current_message = (user_message or "").strip()
        for row in reversed(rows):
            if str(row.get("Who", "")).strip() != "A":
                continue

            action_name = str(row.get("Action", "")).strip()
            if action_name not in {"Offer", "SubAgreement", "Accept"}:
                continue

            logged_message = (row.get("Message", "") or "").strip()
            if current_message and logged_message and logged_message != current_message:
                continue

            utility = row.get("AgentAUtility", None)
            if utility is None:
                return {"utility": None}
            try:
                return {"utility": float(utility)}
            except Exception:
                return {"utility": None}

        return {"utility": None}
    except Exception as e:
        print(f"[{session_id}] Human utility display lookup failed: {e}")
        return {"utility": None}


def _append_utility_chip(message: str, utility: float, scope_issues: list, language: str) -> str:
    """Append a compact utility bar below opponent message."""
    if utility is None:
        return message

    utility_pct = max(0.0, min(100.0, utility * 100.0))
    label = "Skor" if language == "Türkçe" else "Utility"
    scope_label = "Konusulan Konular" if language == "Türkçe" else "Active Issues"
    scope_text = ", ".join(scope_issues) if scope_issues else "-"
    bar_html = (
        "<div style=\"margin-top:6px;max-width:220px;background:#111827;border:1px solid #374151;"
        "border-radius:8px;padding:6px 8px;\">"
        f"<div style=\"display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#9ca3af;"
        f"margin-bottom:4px;\"><span>{label}</span><span style=\"color:#6ee7b7;font-weight:700;\">{utility_pct:.1f}%</span></div>"
        "<div style=\"height:6px;background:#374151;border-radius:999px;overflow:hidden;\">"
        f"<div style=\"height:100%;width:{utility_pct:.1f}%;background:linear-gradient(90deg,#059669 0%,#10b981 100%);\"></div>"
        "</div>"
        f"<div style=\"margin-top:6px;font-size:11px;line-height:1.25;color:#9ca3af;\"><span style=\"color:#d1d5db;\">{scope_label}:</span> {scope_text}</div>"
        "</div>"
    )

    if not message:
        return bar_html
    return f"{message}\n\n{bar_html}"


def _append_lock_sync_marker(message: str, locked_json: str) -> str:
    """Embed lock sync payload in assistant message as a hidden marker."""
    payload = locked_json if locked_json else "{}"
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    marker = (
        f"<span class=\"offer-lock-sync-marker\" data-lock-b64=\"{encoded}\">{encoded}</span>"
    )
    if not message:
        return marker
    return f"{message}\n\n{marker}"


def _append_user_utility_sync_marker(message: str, utility: float, user_message: str) -> str:
    """Embed latest human-action utility so frontend can sync user chat bubble precisely."""
    if utility is None:
        return message

    payload = {
        "utility": float(utility),
        "message": (user_message or "").strip(),
        "ts": time.time(),
    }
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
    marker = (
        f"<span class=\"offer-user-utility-sync-marker\" data-user-util-b64=\"{encoded}\">{encoded}</span>"
    )
    if not message:
        return marker
    return f"{message}\n\n{marker}"


def get_domain_profile(domain: str, agent_type: str, language: str) -> str:
    """
    Get profile path for domain.

    Args:
        domain: Domain name (always "holiday")
        agent_type: Agent type ("human" or "llm")
        language: Language selection ("English" or "Türkçe")

    Returns:
        Path to profile JSON file
    """
    # Map agent type to profile file (profileA = human, profileB = llm)
    profile_file = "profileA.json" if agent_type == "human" else "profileB.json"
    
    # Map language to directory (Türkçe -> turkisch, English -> englisch)
    language_dir = "turkisch" if language == "Türkçe" else "englisch"
    
    return f"main/domains/{language_dir}/{domain}/{profile_file}"


def initialize_session(state: dict):
    """
    Initialize next negotiation session in sequence.

    Args:
        state: Per-user state dictionary

    Returns:
        Session ID
    """
    if state["session_started"]:
        return state["session_id"]

    domain, agent_type = state["negotiations"][state["current_negotiation_index"]]
    state["current_agent_type"] = agent_type

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Session ID format: {username}_{domain}_{agent_type}_{datetime}
    session_id = f"{state['username']}_{domain}_{agent_type}_{datetime_str}"
    print(f"Initialized session: {session_id}")

    # Create queues for this session (stored globally, not in gr.State)
    user_queues[session_id] = {
        "input": queue.Queue(),
        "output": queue.Queue(),
        "completion": queue.Queue(),  # For negotiation completion signals
        "start_time": time.time()     # Track when negotiation started
    }

    print(f"[{session_id}] Session starting in background thread...")

    # Preferences are MANDATORY - no fallback to default profiles
    if domain not in state.get("user_profiles", {}):
        raise ValueError(
            f"No user profile found for domain {domain}. "
            f"Preferences must be filled before negotiation can begin."
        )

    human_profile = state["user_profiles"][domain]
    print(f"Using user profile: {human_profile}")

    # Use inverted profile (no fallback to default profileB.json)
    if domain not in state.get("llm_profiles", {}):
        raise ValueError(
            f"No inverted LLM profile found for domain {domain}. "
            f"This should have been generated during preference elicitation. "
            f"Please ensure preferences were saved successfully."
        )

    llm_profile = state["llm_profiles"][domain]
    print(f"Using inverted LLM profile: {llm_profile}")

    def on_error(error: Exception):
        """Handle session errors."""
        if session_id in user_queues:
            user_queues[session_id]["output"].put(f"Error: {str(error)}")
        db_service.update_session_status(
            session_id,
            "failed",
            str(error)
        )

    def on_complete(result=None):
        """Handle session completion."""
        # Signal completion via queue
        if session_id in user_queues:
            user_queues[session_id]["completion"].put({
                "completed": True,
                "result": result  # Include TournamentResults for status message
            })

        db_service.update_session_status(
            session_id,
            "completed"
        )

    try:
        negotiation_service.start_session(
            session_id=session_id,
            human_profile_path=human_profile,
            llm_profile_path=llm_profile,
            llm_model_name=config.model.model_name,
            get_human_input=lambda: get_human_input_callback(session_id),
            on_llm_message=lambda msg: on_llm_message_callback(session_id, msg),
            agent_type=agent_type,
            language=state.get("language"),
            on_complete=on_complete,
            on_error=on_error
        )

        # Record session in database
        log_path = negotiation_service._get_log_path(session_id, config.model.model_name)
        db_service.create_session(
            session_id=session_id,
            user_id=state["user_id"],
            llm_model_name=config.model.model_name,
            human_profile_path=human_profile,
            llm_profile_path=llm_profile,
            log_path=log_path
        )
    except Exception:
        # Roll back frontend state so chat cannot proceed with a phantom session_id.
        if session_id in user_queues:
            del user_queues[session_id]
        state["session_started"] = False
        state["session_id"] = None
        raise

    state["session_started"] = True
    state["session_id"] = session_id
    return session_id


def update_progress_text(state: dict) -> gr.update:
    """
    Generate progress text for current negotiation.

    Args:
        state: Per-user state dictionary

    Returns:
        Gradio update for progress markdown
    """
    current = state["current_negotiation_index"] + 1  # 1-indexed for display
    text = f"## Session {current} / 2"
    return gr.update(value=text)


def check_negotiation_status(state: dict):
    """
    Check if current negotiation has completed.

    Args:
        state: Per-user state dictionary

    Returns:
        Tuple of (continue_btn_update, status_text_update, state, timer_update, chat_row_update)
    """
    session_id = state.get("session_id")

    if not session_id or session_id not in user_queues:
        return gr.update(), gr.update(), state, gr.update(), gr.update()

    # Check completion queue (non-blocking)
    try:
        completion_info = user_queues[session_id]["completion"].get_nowait()

        if completion_info.get("completed"):
            # Negotiation completed - show continue button and hide chat
            result = completion_info.get("result", {})
            tournament_results = result.get("TournamentResults", {}) if result else {}

            # Determine completion type
            if tournament_results.get("Result") == "Acceptance":
                status_msg = "✅ **Agreement Reached!** / **Anlaşma Sağlandı!**\n\nClick continue to proceed to the next negotiation."
            else:
                status_msg = "⏱️ **Time Expired** / **Süre Doldu**\n\nClick continue to proceed to the next negotiation."

            return (
                gr.update(visible=True, variant="primary", value="Continue (Devam Et)"),   # Show continue button
                gr.update(value=status_msg, visible=True),  # Show status
                state,
                gr.update(), # No timer update needed on completion
                gr.update(visible=False) # Hide chat
            )

    except queue.Empty:
        # No completion yet - keep waiting
        pass

    # Check if deadline exceeded (timer hit 0)
    start_time = state.get("negotiation_start_time")
    if start_time:
        elapsed = time.time() - start_time
        deadline_seconds = config.session.deadline_time  # 900 seconds = 15 minutes

        if elapsed >= deadline_seconds:
            # Deadline exceeded - show continue button and hide chat
            status_msg = "⏱️ **Time Expired - No Agreement Reached** / **Süre Doldu - Anlaşma Sağlanamadı**\n\nClick continue to proceed to the next negotiation."

            return (
                gr.update(visible=True, variant="primary", value="Continue (Devam Et)"),
                gr.update(value=status_msg, visible=True),
                state,
                gr.update(),  # Keep timer at 0:00
                gr.update(visible=False) # Hide chat
            )

    # Still negotiating - keep chat visible

    # CALCULATE TIMER UPDATE
    timer_html = update_timer_html(state.get("negotiation_start_time"))

    return gr.update(), gr.update(), state, gr.update(value=timer_html), gr.update()



def handle_preference_submit(
    state: dict,
    domain: str,
    *slider_values
):
    """
    Process and save user preferences.

    Args:
        state: User state
        domain: Current domain
        *slider_values: All slider values (issue weights followed by value utilities)

    Returns:
        Tuple of UI updates
    """
    try:
        # Get language and profile structure info
        language = state.get("language")
        language_dir = "turkisch" if language == "Türkçe" else "englisch"

        # Read profileA to get issue structure
        profile_path = f"main/domains/{language_dir}/{domain}/profile.json"
        with open(profile_path, 'r', encoding='utf-8') as f:
            template_profile = json.load(f)

        issue_names = list(template_profile["issueWeights"].keys())
        num_issues = len(issue_names)

        # Issue weights are fixed — read directly from template profile
        issue_weights_decimal = [template_profile["issueWeights"][name] for name in issue_names]

        # Extract value utilities (holiday domain only)
        if domain != "holiday":
            raise ValueError(f"Unknown domain: {domain}. Expected 'holiday'.")

        value_utilities_flat = list(slider_values[num_issues:])
        value_utilities_decimal = [u / 100.0 for u in value_utilities_flat]

        # Build profile data structure
        profile_data = {
            "reservationValue": 0,
            "issueWeights": {},
            "issues": {}
        }

        # Fill issue weights
        for i, issue_name in enumerate(issue_names):
            profile_data["issueWeights"][issue_name] = issue_weights_decimal[i]

        # Fill value utilities from user input
        value_idx = 0
        for issue_name in issue_names:
            profile_data["issues"][issue_name] = {}
            value_names = list(template_profile["issues"][issue_name].keys())
            for value_name in value_names:
                profile_data["issues"][issue_name][value_name] = value_utilities_decimal[value_idx]
                value_idx += 1

        # Save via backend
        profile_path = negotiation_service.save_user_profile(
            username=state["username"],
            language=state["language"],
            domain=domain,
            profile_data=profile_data
        )

        # Store in state
        state["user_profiles"][domain] = profile_path

        # Generate inverted LLM opponent profile using Mixed method
        llm_profile_path = negotiation_service.generate_inverted_profile(
            username=state["username"],
            language=state["language"],
            domain=domain,
            human_profile_data=profile_data
        )

        # Store LLM profile path in state for session initialization
        state["llm_profiles"][domain] = llm_profile_path

        # Generate and save bid space plot alongside profiles
        from main.nenv.Preference import Preference
        from main.nenv.BidSpace import BidSpace

        pref_user = Preference(profile_path)
        pref_llm = Preference(llm_profile_path)
        bid_space = BidSpace(pref_user, pref_llm)

        points = bid_space.bid_points
        xs = [bp.utility_a for bp in points]
        ys = [bp.utility_b for bp in points]

        pareto = sorted(bid_space.pareto, key=lambda bp: bp.utility_a)
        px = [bp.utility_a for bp in pareto]
        py = [bp.utility_b for bp in pareto]

        nash = bid_space.nash_point
        kalai = bid_space.kalai_point
        nash_label = f"Nash (H={nash.utility_a:.2f} A={nash.utility_b:.2f})"
        kalai_label = f"Kalai (H={kalai.utility_a:.2f} A={kalai.utility_b:.2f})"

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(xs, ys, s=8, alpha=0.4, label=f"All bids (n={len(points)})")
        ax.plot(px, py, "-*k", label="Pareto frontier")
        ax.plot(nash.utility_a, nash.utility_b, "^r", markersize=10, label=nash_label)
        ax.plot(kalai.utility_a, kalai.utility_b, "or", markersize=10, label=kalai_label)
        opposition = math.sqrt((1 - kalai.utility_a) ** 2 + (1 - kalai.utility_b) ** 2) / math.sqrt(2)

        ax.set_xlabel("Human utility")
        ax.set_ylabel("Agent utility")
        ax.set_title(f"Bid Space — {len(points)} bids")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(f"Opposition: {opposition:.3f}")
        ax.legend(handles, labels, frameon=False, labelspacing=0.8)

        plot_path = os.path.join(os.path.dirname(profile_path), "bid_space.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        success_msg = "✓ Preferences saved!" if language == "English" else "✓ Tercihler kaydedildi!"

        return (
            success_msg,
            gr.update(visible=False),  # Hide preference page
            state
        )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"Error saving preferences: {e}")
        import traceback
        traceback.print_exc()
        return (
            error_msg,
            gr.update(),  # Don't hide preference page
            state
        )


def handle_pre_survey_complete_transition(state: dict):
    """
    Handle pre-survey completion and transition to holiday preference page.

    Args:
        state: Per-user state dictionary

    Returns:
        Tuple of updates for UI components
    """
    # Generate negotiation order (2 holiday negotiations)
    state["negotiations"] = generate_negotiation_order()
    state["current_negotiation_index"] = 0

    # Show holiday preference page for the user's language
    language = state.get("language")
    holiday_en_visible = (language == "English")
    holiday_tr_visible = (language == "Türkçe")

    return (
        "",  # Error message
        gr.update(visible=False),  # Hide pre-survey page
        gr.update(visible=holiday_en_visible),  # Holiday EN pref page
        gr.update(visible=holiday_tr_visible),  # Holiday TR pref page
        state
    )




def chat(message: str, history, state: dict, offer_json: str, lock_json: str):
    """
    Handle chat messages from user.

    Args:
        message: User's message
        history: Chat history
        state: Per-user state dictionary
        offer_json: JSON string of currently selected offer values, e.g. '{"Destination": "Paris"}'
        lock_json: Current locked-issues JSON (unused input, kept for UI sync channel)

    Returns:
        Tuple of (LLM response, reset offer_json, locked offer_json for panel sync)
    """
    if not message.strip():
        return "Please enter a message.", offer_json, "{}"

    session_id = state["session_id"]
    if not session_id:
        return "Error: Session not initialized.", offer_json, "{}"

    # Ensure backend session exists before touching queues/timers.
    try:
        negotiation_service.get_session_status(session_id)
    except Exception as e:
        return f"Error: Backend session not found ({e}). Please restart this negotiation.", offer_json, "{}"

    # Start the deadline timer on first message (not when page loads)
    if state.get("negotiation_start_time") is None:
        state["negotiation_start_time"] = time.time()
        try:
            negotiation_service.start_negotiation_timer(session_id)
            print(f"[{session_id}] User sent first message - deadline timer started")
        except Exception as e:
            print(f"[{session_id}] Error starting timer: {e}")
            return f"Error: Could not start negotiation timer ({e}). Please restart this negotiation.", offer_json, "{}"

    # Check if queues exist for this session
    if session_id not in user_queues:
        return "Error: Session not properly initialized", offer_json, "{}"

    # Parse offer selection and locked issues from hidden panel state.
    bid_dict = json.loads(offer_json) if offer_json and offer_json.strip() not in ("", "{}") else None
    locked_dict = json.loads(lock_json) if lock_json and lock_json.strip() not in ("", "{}") else {}

    # Send user message + panel payload (unblocks get_human_input_callback)
    print(
        f"[{session_id}] User message queued (bid_dict={bid_dict}, locked_dict={locked_dict}), "
        "waiting for LLM response..."
    )
    user_queues[session_id]["input"].put((message, bid_dict, locked_dict))

    # Wait for LLM response
    try:
        response = user_queues[session_id]["output"].get(timeout=120)
        print(f"[{session_id}] LLM response received")
        response_text = response if isinstance(response, str) else str(response)
        display_data = _get_latest_opponent_display_data(session_id, response_text)
        human_display_data = _get_latest_human_display_data(session_id, message)
        rendered_response = _append_utility_chip(
            response_text,
            display_data.get("utility"),
            display_data.get("scope_issues", []),
            state.get("language", "English")
        )
        rendered_response = _append_user_utility_sync_marker(
            rendered_response,
            human_display_data.get("utility"),
            message
        )
        locked_json = _get_locked_agreements_json_for_offer_panel(session_id)
        rendered_response = _append_lock_sync_marker(rendered_response, locked_json)
        print(f"[{session_id}] lock_sync payload: {locked_json}")
        return rendered_response, "{}", locked_json
    except queue.Empty:
        return "Error: Timeout waiting for response", offer_json, "{}"


# Build UI
with gr.Blocks(
    title="Negotiation Chat",
    css="""
    .accept-bridge-hidden {
        display: none !important;
    }
    .message.bot,
    .message.assistant {
        background: #374151 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 10px !important;
    }
    .message.bot .message-content,
    .message.assistant .message-content {
        background: transparent !important;
        color: #e5e7eb !important;
        border: none !important;
        width: 100% !important;
    }
    .message.bot .message-content > *,
    .message.assistant .message-content > * {
        background: transparent !important;
    }
    .offer-lock-sync-marker {
        display: none !important;
    }
    .offer-user-utility-sync-marker {
        display: none !important;
    }
    #video-explanation-continue-btn {
        display: none !important;
    }
    #video-explanation-continue-btn.video-ready {
        display: block !important;
    }
    """,
    theme=gr.themes.Soft(
        primary_hue="violet",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"],
        text_size="md",
        radius_size="sm"   # less bubbly
    ),
    js="""
    () => {
        const url = new URL(window.location.href);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """,
    head='''
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
window.videoExplanationPlayer = null;

window.onYouTubeIframeAPIReady = function() {
    window.videoExplanationApiReady = true;
};

window.setVideoExplanationContinueState = function(isReady) {
    var wrapper = document.getElementById('__VIDEO_EXPLANATION_CONTINUE_BUTTON_ID__');
    if (!wrapper) return;

    var button = wrapper.tagName === 'BUTTON' ? wrapper : wrapper.querySelector('button');
    if (button) {
        button.disabled = !isReady;
    }

    if (isReady) {
        wrapper.classList.add('video-ready');
    } else {
        wrapper.classList.remove('video-ready');
    }
};

window.initVideoExplanation = function() {
    window.setVideoExplanationContinueState(false);

    var iframe = document.getElementById('video-explanation-player');
    if (!iframe) return;

    if (!window.YT || !window.YT.Player || !window.videoExplanationApiReady) {
        setTimeout(window.initVideoExplanation, 250);
        return;
    }

    if (window.videoExplanationPlayer && typeof window.videoExplanationPlayer.destroy === 'function') {
        try {
            window.videoExplanationPlayer.destroy();
        } catch (err) {
            console.warn('Failed to reset video explanation player', err);
        }
    }

    window.videoExplanationPlayer = new window.YT.Player('video-explanation-player', {
        events: {
            onStateChange: function(event) {
                if (event.data === window.YT.PlayerState.ENDED) {
                    window.setVideoExplanationContinueState(true);
                }
            }
        }
    });
};

// Offer panel initializer — called via <img onload> when panel HTML is injected.
// Scripts inside innerHTML are not executed by browsers, so we use this trick.
window.initOfferPanel = function(pageIdx) {
    var panelId = 'offer_panel_' + pageIdx;
    var tbId = 'offer_tb_' + pageIdx;
    var lockTbId = 'offer_lock_tb_' + pageIdx;
    var sendOfferMsgId = 'send_offer_msg_' + pageIdx;
    var chatInputId = 'chat_input_' + pageIdx;
    var acceptBridgeId = 'accept_bridge_' + pageIdx;
    var sendOfferBridgeId = 'send_offer_bridge_' + pageIdx;
    var utilDataId = 'offer_utility_data_' + pageIdx;
    var _sel = {};
    var _locked = {};
    var _utilityModel = null;
    var _sendingFromOfferButton = false;
    var _isClientOfferTextboxWrite = false;

    (function loadUtilityModel() {
        var dataEl = document.getElementById(utilDataId);
        if (!dataEl) return;
        try {
            _utilityModel = JSON.parse(dataEl.textContent || '{}');
        } catch (err) {
            console.warn('Failed to parse offer utility model for page', pageIdx, err);
            _utilityModel = null;
        }
    })();

    function _setTb(val) {
        var el = document.getElementById(tbId);
        if (!el) return;
        var ta = el.querySelector('textarea, input');
        if (!ta) return;
        _isClientOfferTextboxWrite = true;
        var proto = ta.tagName === 'TEXTAREA'
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
        setter.call(ta, val);
        ta.dispatchEvent(new Event('input', {bubbles: true}));
        setTimeout(function() {
            _isClientOfferTextboxWrite = false;
        }, 0);
    }

    function _getOfferTextboxTextarea() {
        var el = document.getElementById(tbId);
        if (!el) return null;
        return el.querySelector('textarea, input');
    }

    function _getLockTextboxTextarea() {
        var el = document.getElementById(lockTbId);
        if (!el) return null;
        return el.querySelector('textarea, input');
    }

    function _setSendOfferMessageTb(val) {
        var el = document.getElementById(sendOfferMsgId);
        if (!el) return;
        var ta = el.querySelector('textarea, input');
        if (!ta) return;
        var proto = ta.tagName === 'TEXTAREA'
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
        setter.call(ta, val);
        ta.dispatchEvent(new Event('input', {bubbles: true}));
    }

    function _triggerAcceptBridge() {
        var bridge = document.getElementById(acceptBridgeId);
        if (!bridge) return;
        if (bridge.tagName === 'BUTTON' && !bridge.disabled) {
            bridge.click();
            return;
        }
        var btn = bridge.querySelector('button');
        if (btn && !btn.disabled) {
            btn.click();
            return;
        }
        if (typeof bridge.click === 'function') {
            bridge.click();
        }
    }

    function _triggerSendOfferBridge() {
        var bridge = document.getElementById(sendOfferBridgeId);
        if (!bridge) return;
        if (bridge.tagName === 'BUTTON' && !bridge.disabled) {
            bridge.click();
            return;
        }
        var btn = bridge.querySelector('button');
        if (btn && !btn.disabled) {
            btn.click();
            return;
        }
        if (typeof bridge.click === 'function') {
            bridge.click();
        }
    }

    function _getChatTextarea() {
        var wrap = document.getElementById(chatInputId);
        if (wrap) {
            if (wrap.tagName === 'TEXTAREA') return wrap;
            var ta = wrap.querySelector('textarea');
            if (ta && ta.offsetParent !== null) return ta;
        }
        var all = document.querySelectorAll('textarea');
        for (var i = 0; i < all.length; i++) {
            var ta2 = all[i];
            if (ta2.closest && ta2.closest('#' + tbId)) continue;
            if (ta2.offsetParent === null || ta2.disabled || ta2.readOnly) continue;
            return ta2;
        }
        return null;
    }

    function _getVisibleUserMessageNodes() {
        var selectors = [
            '.message.user .message-content',
            '.message.user'
        ];

        for (var i = 0; i < selectors.length; i++) {
            var nodes = document.querySelectorAll(selectors[i]);
            var visible = [];
            for (var j = 0; j < nodes.length; j++) {
                var node = nodes[j];
                if (node && node.offsetParent !== null) {
                    visible.push(node);
                }
            }
            if (visible.length > 0) {
                return visible;
            }
        }

        return [];
    }

    function _normalizeMessageText(raw) {
        return String(raw || '').replace(/\\s+/g, ' ').trim();
    }

    function _getMessageNodeText(node) {
        if (!node) return '';
        var clone = node.cloneNode(true);
        if (clone.querySelectorAll) {
            clone.querySelectorAll('.user-offer-utility-mini').forEach(function(el) {
                if (el && el.parentNode) {
                    el.parentNode.removeChild(el);
                }
            });
        }
        return _normalizeMessageText(clone.textContent || '');
    }

    function _findLatestVisibleUserMessageNodeByText(messageText) {
        var nodes = _getVisibleUserMessageNodes();
        if (!nodes.length) return null;

        var needle = _normalizeMessageText(messageText);
        if (!needle) {
            return nodes[nodes.length - 1];
        }

        for (var i = nodes.length - 1; i >= 0; i--) {
            var nodeText = _getMessageNodeText(nodes[i]);
            if (!nodeText) continue;
            if (nodeText === needle || nodeText.indexOf(needle) !== -1) {
                return nodes[i];
            }
        }

        return null;
    }

    function _buildMiniUtilityBarHtml(label, utilityPct) {
        return (
            '<div class="user-offer-utility-mini" style="margin-top:6px;max-width:220px;background:#111827;border:1px solid #374151;border-radius:8px;padding:6px 8px;">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#9ca3af;margin-bottom:4px;">' +
                    '<span>' + label + '</span>' +
                    '<span style="color:#6ee7b7;font-weight:700;">' + utilityPct.toFixed(1) + '%</span>' +
                '</div>' +
                '<div style="height:6px;background:#374151;border-radius:999px;overflow:hidden;">' +
                    '<div style="height:100%;width:' + utilityPct.toFixed(1) + '%;background:linear-gradient(90deg,#059669 0%,#10b981 100%);"></div>' +
                '</div>' +
            '</div>'
        );
    }

    function _upsertMiniUtilityBar(node, label, utilityPct) {
        if (!node || !node.querySelector) return false;
        var wrapper = document.createElement('div');
        wrapper.innerHTML = _buildMiniUtilityBarHtml(label, utilityPct);
        var fresh = wrapper.firstChild;
        if (!fresh) return false;

        var existing = node.querySelector('.user-offer-utility-mini');
        if (existing && existing.parentNode) {
            existing.parentNode.replaceChild(fresh, existing);
        } else {
            node.appendChild(fresh);
        }
        return true;
    }

    function _annotateLatestMatchingUserMessageWithUtility(utilityPct, messageText) {
        var panel = document.getElementById(panelId);
        var label = panel ? (panel.getAttribute('data-user-utility-label') || 'Utility') : 'Utility';
        var pct = Math.max(0, Math.min(100, Number(utilityPct) || 0));
        var attempts = 0;
        var maxAttempts = 18;

        function tryAttach() {
            attempts += 1;
            var node = _findLatestVisibleUserMessageNodeByText(messageText);
            if (node) {
                if (_upsertMiniUtilityBar(node, label, pct)) {
                    return;
                }
            }
            if (attempts < maxAttempts) {
                setTimeout(tryAttach, 90);
            }
        }

        setTimeout(tryAttach, 80);
    }

    function _annotateNextUserMessageWithUtility(utilityPct, messageText) {
        var panel = document.getElementById(panelId);
        var label = panel ? (panel.getAttribute('data-user-utility-label') || 'Utility') : 'Utility';
        var pct = Math.max(0, Math.min(100, Number(utilityPct) || 0));
        var baselineCount = _getVisibleUserMessageNodes().length;
        var attempts = 0;
        var maxAttempts = 24;

        function tryAttach() {
            attempts += 1;
            var nodes = _getVisibleUserMessageNodes();
            if (nodes.length > baselineCount) {
                var node = _findLatestVisibleUserMessageNodeByText(messageText) || nodes[nodes.length - 1];
                if (node && _upsertMiniUtilityBar(node, label, pct)) {
                    return;
                }
            }
            if (attempts < maxAttempts) {
                setTimeout(tryAttach, 90);
            }
        }

        setTimeout(tryAttach, 80);
    }

    function _computeUtility(selection) {
        if (!_utilityModel || !_utilityModel.issueWeights || !_utilityModel.issues) return 0;
        var total = 0;
        var issueWeights = _utilityModel.issueWeights;
        var issueValues = _utilityModel.issues;

        Object.keys(selection).forEach(function(issue) {
            var selectedValue = selection[issue];
            var weight = Number(issueWeights[issue] || 0);
            var valueUtility = 0;

            if (
                issueValues[issue] &&
                Object.prototype.hasOwnProperty.call(issueValues[issue], selectedValue)
            ) {
                valueUtility = Number(issueValues[issue][selectedValue] || 0);
            }

            total += weight * valueUtility;
        });

        return total;
    }

    function _applySelectionAndLockStyles() {
        var panel = document.getElementById(panelId);
        if (!panel) return;

        panel.querySelectorAll('.offer-issue-row').forEach(function(row) {
            var lbl = row.querySelector('.offer-issue-label');
            if (!lbl) return;
            var issue = lbl.textContent.trim();
            var issueLocked = Object.prototype.hasOwnProperty.call(_locked, issue);
            var selectedValue = Object.prototype.hasOwnProperty.call(_sel, issue) ? String(_sel[issue]) : null;
            var lockedValue = issueLocked ? String(_locked[issue]) : null;

            row.classList.toggle('locked', issueLocked);
            row.querySelectorAll('.offer-val-btn').forEach(function(btn) {
                var btnValue = (btn.textContent || '').trim();
                var isSelected = selectedValue !== null && btnValue === selectedValue;
                var isLockedValue = issueLocked && lockedValue === btnValue;

                btn.classList.toggle('selected', isSelected);
                btn.classList.toggle('locked', isLockedValue);
                btn.classList.toggle('disabled', issueLocked);
                btn.disabled = issueLocked;
            });
        });
    }

    function _applyLockMap(parsed) {
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return;
        console.debug('[offer-lock]', pageIdx, parsed);
        _locked = Object.assign({}, parsed);
        _sel = Object.assign({}, _locked);
        _setTb(JSON.stringify(_sel));
        _applySelectionAndLockStyles();
        _updateUtilityDisplay();
    }

    function _updateUtilityDisplay() {
        var panel = document.getElementById(panelId);
        if (!panel) return;

        var fillEl = panel.querySelector('.offer-utility-fill');
        var valueEl = panel.querySelector('.offer-utility-value');
        var metaEl = panel.querySelector('.offer-utility-meta');
        var utilWrap = panel.querySelector('.offer-utility-wrap');
        if (!fillEl || !valueEl || !metaEl || !utilWrap) return;

        var utility = _computeUtility(_sel);
        var percentage = Math.max(0, Math.min(100, utility * 100));
        fillEl.style.width = percentage.toFixed(1) + '%';
        valueEl.textContent = percentage.toFixed(1) + '%';

        var totalIssues = (_utilityModel && _utilityModel.issueWeights)
            ? Object.keys(_utilityModel.issueWeights).length
            : panel.querySelectorAll('.offer-issue-row').length;
        var selectedIssues = Object.keys(_sel).length;
        var metaLabel = utilWrap.getAttribute('data-utility-meta-label') || 'issues selected';
        metaEl.textContent = selectedIssues + '/' + totalIssues + ' ' + metaLabel;
    }

    window['offerSelectValue_' + pageIdx] = function(issue, value, btn) {
        if (Object.prototype.hasOwnProperty.call(_locked, issue)) return;

        var panel = document.getElementById(panelId);
        panel.querySelectorAll('.offer-issue-row').forEach(function(row) {
            var lbl = row.querySelector('.offer-issue-label');
            if (lbl && lbl.textContent.trim() === issue) {
                row.querySelectorAll('.offer-val-btn').forEach(function(b) {
                    b.classList.remove('selected');
                });
            }
        });
        if (_sel[issue] === value) {
            delete _sel[issue];
        } else {
            btn.classList.add('selected');
            _sel[issue] = value;
        }
        _setTb(JSON.stringify(_sel));
        _applySelectionAndLockStyles();
        _updateUtilityDisplay();
    };

    window['offerClear_' + pageIdx] = function() {
        _sel = Object.assign({}, _locked);
        _setTb(JSON.stringify(_sel));
        _applySelectionAndLockStyles();
        _updateUtilityDisplay();
    };

    window['offerAgree_' + pageIdx] = function() {
        _triggerAcceptBridge();
    };

    window['offerSendOffer_' + pageIdx] = function() {
        var selectedBid = _sel || {};
        if (!_hasOfferDelta(selectedBid)) return;

        var chatTa = _getChatTextarea();
        if (!chatTa) return;
        var messageText = (chatTa.value || '').trim();
        if (!messageText) return;

        var utilityPct = Math.max(0, Math.min(100, _computeUtility(selectedBid) * 100));
        _setTb(JSON.stringify(selectedBid));
        _setSendOfferMessageTb(messageText);
        _annotateNextUserMessageWithUtility(utilityPct, messageText);

        _sendingFromOfferButton = true;
        setTimeout(function() {
            _triggerSendOfferBridge();
            var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            setter.call(chatTa, '');
            chatTa.dispatchEvent(new Event('input', {bubbles: true}));
            _sendingFromOfferButton = false;
        }, 40);
    };

    window.__offerPanelRuntime = window.__offerPanelRuntime || {};
    window.__offerPanelRuntime[pageIdx] = {
        getSelection: function() { return _sel; },
        clearSelection: function() { window['offerClear_' + pageIdx](); }
    };

    function _hasOfferDelta(selection) {
        var keys = Object.keys(selection || {});
        if (keys.length === 0) return false;

        for (var i = 0; i < keys.length; i++) {
            var issue = keys[i];
            if (!Object.prototype.hasOwnProperty.call(_locked, issue)) return true;
            if (String(selection[issue]) !== String(_locked[issue])) return true;
        }

        return Object.keys(_locked).length === 0;
    }

    function _bindSendHook() {
        var hookKey = '__offerSendHookBound_' + pageIdx;
        if (window[hookKey]) return;
        window[hookKey] = true;

        // Handle Enter-to-send in chat textbox.
        document.addEventListener('keydown', function(evt) {
            if (!evt || evt.key !== 'Enter' || evt.shiftKey) return;
            var target = evt.target;
            if (!target || target.tagName !== 'TEXTAREA') return;
            if (target.closest && target.closest('#' + tbId)) return;
            if (target.offsetParent === null || target.disabled || target.readOnly) return;
            if (_sendingFromOfferButton) return;

            var runtime = window.__offerPanelRuntime && window.__offerPanelRuntime[pageIdx];
            if (!runtime || typeof runtime.getSelection !== 'function') return;
            var selectedBid = runtime.getSelection() || {};
            if (!_hasOfferDelta(selectedBid)) return;

            var messageText = (target.value || '').trim();
            if (!messageText) return;

            // Re-sync the hidden Gradio textbox immediately before native submit so
            // the backend receives the latest panel-backed offer payload.
            _setTb(JSON.stringify(selectedBid));

            var utilityPct = Math.max(0, Math.min(100, _computeUtility(selectedBid) * 100));
            _annotateNextUserMessageWithUtility(utilityPct, messageText);
        }, true);
    }

    function _bindOfferTextboxSync() {
        var syncKey = '__offerTextboxSyncBound_' + pageIdx;
        if (window[syncKey]) return;

        function tryBind() {
            var offerTa = _getOfferTextboxTextarea();
            if (!offerTa) {
                setTimeout(tryBind, 120);
                return;
            }

            window[syncKey] = true;
            var lastRaw = null;

            function applyOfferRaw(raw) {
                raw = (raw || '').trim();
                if (!raw) raw = '{}';
                if (raw === lastRaw) return;
                lastRaw = raw;

                if (_isClientOfferTextboxWrite) return;

                if (raw === '{}') {
                    _sel = Object.assign({}, _locked);
                    _applySelectionAndLockStyles();
                    _updateUtilityDisplay();
                    return;
                }

                try {
                    var parsed = JSON.parse(raw);
                    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return;
                    _sel = Object.assign({}, parsed);
                    _applySelectionAndLockStyles();
                    _updateUtilityDisplay();
                } catch (_) {
                    return;
                }
            }

            offerTa.addEventListener('input', function() {
                applyOfferRaw(offerTa.value);
            });

            setInterval(function() {
                applyOfferRaw(offerTa.value);
            }, 220);
        }

        tryBind();
    }

    function _bindLockTextboxSync() {
        var syncKey = '__offerLockTextboxSyncBound_' + pageIdx;
        if (window[syncKey]) return;

        function tryBind() {
            var lockTa = _getLockTextboxTextarea();
            if (!lockTa) {
                setTimeout(tryBind, 120);
                return;
            }

            window[syncKey] = true;
            var lastRaw = null;

            function applyLockRaw(raw) {
                raw = (raw || '').trim();
                if (!raw) raw = '{}';
                if (raw === lastRaw) return;
                lastRaw = raw;

                try {
                    var parsed = JSON.parse(raw);
                    _applyLockMap(parsed);
                } catch (_) {
                    return;
                }
            }

            lockTa.addEventListener('input', function() {
                applyLockRaw(lockTa.value);
            });

            setInterval(function() {
                applyLockRaw(lockTa.value);
            }, 220);
        }

        tryBind();
    }

    function _bindChatMarkerLockSync() {
        var syncKey = '__offerChatMarkerLockSyncBound_' + pageIdx;
        if (window[syncKey]) return;
        window[syncKey] = true;
        var lastMarker = '';

        function _decodeBase64Utf8(encoded) {
            var binary = atob(encoded);
            var bytes = new Uint8Array(binary.length);
            for (var i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            if (typeof TextDecoder !== 'undefined') {
                return new TextDecoder('utf-8').decode(bytes);
            }
            // Fallback for older environments
            var escaped = '';
            for (var j = 0; j < bytes.length; j++) {
                escaped += '%' + ('00' + bytes[j].toString(16)).slice(-2);
            }
            return decodeURIComponent(escaped);
        }

        function applyFromLatestMarker() {
            var markers = document.querySelectorAll('.offer-lock-sync-marker');
            for (var i = markers.length - 1; i >= 0; i--) {
                var marker = markers[i];
                if (!marker) continue;

                var msgNode = marker.closest ? marker.closest('.message') : null;
                if (msgNode && msgNode.offsetParent === null) continue;

                var encoded = (marker.getAttribute('data-lock-b64') || marker.textContent || '').trim();
                if (!encoded || encoded === lastMarker) return;

                try {
                    var raw = _decodeBase64Utf8(encoded);
                    var parsed = JSON.parse(raw);
                    _applyLockMap(parsed);
                    lastMarker = encoded;
                } catch (_) {
                    return;
                }
                return;
            }
        }

        setInterval(applyFromLatestMarker, 180);
        setTimeout(applyFromLatestMarker, 80);
    }

    function _bindChatMarkerUserUtilitySync() {
        var syncKey = '__offerChatMarkerUserUtilitySyncBound_' + pageIdx;
        if (window[syncKey]) return;
        window[syncKey] = true;
        var lastMarker = '';

        function _decodeBase64Utf8(encoded) {
            var binary = atob(encoded);
            var bytes = new Uint8Array(binary.length);
            for (var i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            if (typeof TextDecoder !== 'undefined') {
                return new TextDecoder('utf-8').decode(bytes);
            }
            var escaped = '';
            for (var j = 0; j < bytes.length; j++) {
                escaped += '%' + ('00' + bytes[j].toString(16)).slice(-2);
            }
            return decodeURIComponent(escaped);
        }

        function applyFromLatestMarker() {
            var markers = document.querySelectorAll('.offer-user-utility-sync-marker');
            for (var i = markers.length - 1; i >= 0; i--) {
                var marker = markers[i];
                if (!marker) continue;

                var msgNode = marker.closest ? marker.closest('.message') : null;
                if (msgNode && msgNode.offsetParent === null) continue;

                var encoded = (marker.getAttribute('data-user-util-b64') || marker.textContent || '').trim();
                if (!encoded || encoded === lastMarker) return;

                try {
                    var raw = _decodeBase64Utf8(encoded);
                    var parsed = JSON.parse(raw);
                    var utility = Number(parsed.utility);
                    if (!isFinite(utility)) {
                        lastMarker = encoded;
                        return;
                    }
                    _annotateLatestMatchingUserMessageWithUtility(utility * 100, parsed.message);
                    lastMarker = encoded;
                } catch (_) {
                    return;
                }
                return;
            }
        }

        setInterval(applyFromLatestMarker, 180);
        setTimeout(applyFromLatestMarker, 80);
    }

    _bindSendHook();
    _bindOfferTextboxSync();
    _bindLockTextboxSync();
    _bindChatMarkerLockSync();
    _bindChatMarkerUserUtilitySync();
    _applySelectionAndLockStyles();
    _updateUtilityDisplay();
};

window.initDragDrop = function(containerId, bridgeId) {
    function tryInit() {
        var el = document.getElementById(containerId);
        if (!el) { setTimeout(tryInit, 100); return; }
        if (el.dataset.sortableInit === 'done') return;
        el.dataset.sortableInit = 'done';
        
        // Determine if this is an issue container or value container
        var isIssueContainer = containerId.startsWith('issue_sort_');
        
        Sortable.create(el, {
            animation: 150,
            ghostClass: 'sortable-ghost',
            chosenClass: 'sortable-chosen',
            draggable: '.sortable-item',
            forceFallback: true,
            onEnd: function(evt) {
                var items = el.querySelectorAll('.sortable-item');
                var ids = [];
                for (var i = 0; i < items.length; i++) {
                    ids.push(items[i].getAttribute('data-id'));
                }
                
                console.log('Drag ended, container:', containerId, 'new order:', ids);
                
                var bridge = document.getElementById(bridgeId);
                console.log('Bridge element:', bridgeId, bridge);
                if (bridge) {
                    var ta = bridge.querySelector('textarea');
                    if (ta) {
                        // Read existing value and update
                        var currentData = {};
                        try {
                            currentData = JSON.parse(ta.value) || {};
                        } catch(e) {
                            currentData = {};
                        }
                        
                        if (isIssueContainer) {
                            currentData.issues = ids;
                        } else {
                            // It's a value container - extract issue name from container ID
                            // Format: values_{issueName}_{domain}_{lang}
                            var parts = containerId.split('_');
                            if (parts.length >= 2) {
                                var issueName = parts[1]; // Get issue name part
                                if (!currentData.values) currentData.values = {};
                                currentData.values[issueName] = ids;
                            }
                        }
                        
                        var newValue = JSON.stringify(currentData);
                        console.log('Storing order data:', newValue);
                        
                        // Set value using native setter
                        var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                        nativeSetter.call(ta, newValue);
                        
                        ta.dispatchEvent(new Event('input', {bubbles: true}));
                        ta.dispatchEvent(new Event('change', {bubbles: true}));
                        console.log('Events dispatched, value:', ta.value);
                    }
                }
            }
        });
    }
    tryInit();
};
</script>
'''.replace('__VIDEO_EXPLANATION_CONTINUE_BUTTON_ID__', VIDEO_EXPLANATION_CONTINUE_BUTTON_ID)
) as demo:

    # Per-user state (isolated per browser session)
    user_state = gr.State(initialize_user_state())

    # Create pages
    (username_page, username_btn, username_inputs,
     username_outputs, username_handler) = create_username_page(db_service)

    # Create pre-survey page (before negotiations)
    (pre_survey_page, pre_survey_btn, pre_survey_inputs,
     pre_survey_outputs, pre_survey_handler) = create_survey_page(
        survey_url=PRE_SURVEY_LINK,
        title="Pre-Negotiation Survey (Deney Öncesi Anket)",
        subtitle="Please complete the survey below before continuing. / Lütfen müzakereye devam etmeden önce aşağıdaki anketi doldurun.",
        button_text="Continue (Devam Et)"
    )

    # Create preference elicitation pages (2 total: holiday × 2 languages)
    # Holiday - English
    (holiday_pref_page_en, holiday_pref_btn_en, holiday_issue_sliders_en,
     holiday_value_sliders_en, holiday_status_en, holiday_sum_display_en,
     holiday_issue_names_en, holiday_value_names_en, holiday_order_textbox_en) = create_preference_elicitation_page(
        domain="holiday",
        language="English"
    )

    # Holiday - Turkish
    (holiday_pref_page_tr, holiday_pref_btn_tr, holiday_issue_sliders_tr,
     holiday_value_sliders_tr, holiday_status_tr, holiday_sum_display_tr,
     holiday_issue_names_tr, holiday_value_names_tr, holiday_order_textbox_tr) = create_preference_elicitation_page(
        domain="holiday",
        language="Türkçe"
    )

    # Organize preference pages for easy access
    pref_pages = {
        ("holiday", "English"): {
            "page": holiday_pref_page_en,
            "btn": holiday_pref_btn_en,
            "issue_sliders": holiday_issue_sliders_en,
            "value_sliders": holiday_value_sliders_en,
            "status": holiday_status_en,
            "sum_display": holiday_sum_display_en,
            "issue_names": holiday_issue_names_en,
            "value_names": holiday_value_names_en,
            "order_textbox": holiday_order_textbox_en
        },
        ("holiday", "Türkçe"): {
            "page": holiday_pref_page_tr,
            "btn": holiday_pref_btn_tr,
            "issue_sliders": holiday_issue_sliders_tr,
            "value_sliders": holiday_value_sliders_tr,
            "status": holiday_status_tr,
            "sum_display": holiday_sum_display_tr,
            "issue_names": holiday_issue_names_tr,
            "value_names": holiday_value_names_tr,
            "order_textbox": holiday_order_textbox_tr
        },
    }

    # Create preference confirmation pages (one per language)
    (confirm_page_en, confirm_btn_en, confirm_prefs_html_en, confirm_value_note_en) = create_preference_confirmation_page("English")
    (confirm_page_tr, confirm_btn_tr, confirm_prefs_html_tr, confirm_value_note_tr) = create_preference_confirmation_page("Türkçe")

    confirm_pages = {
        "English": {"page": confirm_page_en, "btn": confirm_btn_en, "prefs_html": confirm_prefs_html_en, "value_note": confirm_value_note_en},
        "Türkçe":  {"page": confirm_page_tr, "btn": confirm_btn_tr, "prefs_html": confirm_prefs_html_tr, "value_note": confirm_value_note_tr},
    }

    # Create video explanation page (shown after preference confirmation, before negotiation)
    (video_explanation_page, video_component, video_continue_btn) = create_video_explanation_page(
        video_url_en=VIDEO_EXPLANATION_PATH_EN,
        video_url_tr=VIDEO_EXPLANATION_PATH_TR,
    )

    # Per-session post-surveys (one shown after each negotiation, picked by the
    # agent type the user just faced). The page UI itself is identical and stays
    # generic — agent type is the experiment's independent variable, never shown.
    _per_session_survey_title = "Post Session Survey (Oturum Sonrası Anket)"
    _per_session_survey_subtitle = (
        "Please complete the survey about the negotiation you just completed. / "
        "Lütfen az önce tamamladığınız müzakere hakkındaki anketi doldurun."
    )

    (llm_survey_page, llm_survey_btn, llm_survey_inputs,
     llm_survey_outputs, _llm_survey_handler) = create_survey_page(
        survey_url=LLM_AGENT_SURVEY_LINK,
        title=_per_session_survey_title,
        subtitle=_per_session_survey_subtitle,
        button_text="Continue (Devam Et)"
    )

    (heuristic_survey_page, heuristic_survey_btn, heuristic_survey_inputs,
     heuristic_survey_outputs, _heuristic_survey_handler) = create_survey_page(
        survey_url=HEURISTIC_AGENT_SURVEY_LINK,
        title=_per_session_survey_title,
        subtitle=_per_session_survey_subtitle,
        button_text="Continue (Devam Et)"
    )

    # Create 2 separate negotiation pages for complete isolation between rounds
    (negotiation_page_1, chat_interface_1, progress_text_1,
     continue_btn_1, status_text_1, timer_display_1, preferences_display_1, instructions_html_1, chat_row_1,
     offer_panel_html_1, offer_textbox_1, accept_bridge_btn_1, send_offer_bridge_btn_1, send_offer_msg_tb_1, offer_lock_tb_1) = create_negotiation_page(
        chat_fn=chat,
        config=config,
        user_state=user_state,
        page_idx=1
    )

    (negotiation_page_2, chat_interface_2, progress_text_2,
     continue_btn_2, status_text_2, timer_display_2, preferences_display_2, instructions_html_2, chat_row_2,
     offer_panel_html_2, offer_textbox_2, accept_bridge_btn_2, send_offer_bridge_btn_2, send_offer_msg_tb_2, offer_lock_tb_2) = create_negotiation_page(
        chat_fn=chat,
        config=config,
        user_state=user_state,
        page_idx=2
    )

    negotiation_pages = [negotiation_page_1, negotiation_page_2]
    continue_btns = [continue_btn_1, continue_btn_2]
    status_texts = [status_text_1, status_text_2]
    progress_texts = [progress_text_1, progress_text_2]
    timer_displays = [timer_display_1, timer_display_2]
    preferences_displays = [preferences_display_1, preferences_display_2]
    instructions_htmluctions = [instructions_html_1, instructions_html_2]
    chat_rows = [chat_row_1, chat_row_2]

    (thank_you_page,) = create_thank_you_page()

    # Wire username page events - transitions to pre-survey
    username_btn.click(
        fn=username_handler,
        inputs=username_inputs + [user_state],
        outputs=username_outputs + [username_page, pre_survey_page, user_state]
    )

    # Wire pre-survey page events - transitions to PREFERENCE PAGE
    pre_survey_btn.click(
        fn=handle_pre_survey_complete_transition,
        inputs=[user_state],
        outputs=[
            pre_survey_outputs[0],  # Error message
            pre_survey_page,  # Hide pre-survey
            holiday_pref_page_en,  # Show/hide Holiday EN pref
            holiday_pref_page_tr,  # Show/hide Holiday TR pref
            user_state
        ]
    )

    # Wire preference page buttons - transitions to confirmation page, then negotiation
    def create_pref_submit_handler(domain, language):
        pref_config = pref_pages.get((domain, language), {})
        issue_names = pref_config.get('issue_names', [])
        value_names = pref_config.get('value_names', {})
        confirm_config = confirm_pages[language]

        def handler(state, order_json):
            print(f'[DEBUG] order_json received: {order_json}')

            try:
                order_data = json.loads(order_json) if order_json else {}

                current_issue_order = order_data.get('issues', issue_names)
                if not current_issue_order:
                    current_issue_order = issue_names

                # Calculate issue weights (rightmost = highest) — passed to hidden sliders
                reversed_order = list(reversed(current_issue_order))
                num_issues = len(reversed_order)
                raw_weights = {name: num_issues - i for i, name in enumerate(reversed_order)}
                total = sum(raw_weights.values())
                normalized_weights = {k: (v/total)*100 for k, v in raw_weights.items()}

                calculated_sliders = []
                for name in issue_names:
                    calculated_sliders.append(normalized_weights.get(name, 100/len(issue_names)))

                value_orders = order_data.get('values', {})
                for issue_name in issue_names:
                    values_list = value_names.get(issue_name, [])
                    if not values_list:
                        continue
                    ordered_values = value_orders.get(issue_name, values_list)
                    if not ordered_values:
                        ordered_values = values_list
                    rev_values = list(reversed(ordered_values))
                    num_values = len(rev_values)
                    for v in values_list:
                        if v in rev_values:
                            rank = rev_values.index(v) + 1
                            utility = (num_values - rank + 1) * (100 / num_values)
                        else:
                            raise ValueError(f"Value '{v}' for issue '{issue_name}' not found in DOM order {rev_values}")
                        calculated_sliders.append(utility)

                print(f'[DEBUG] calculated_sliders: {calculated_sliders[:4]}...')

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

            # Save preferences
            status_msg, pref_page_update, new_state = handle_preference_submit(
                state, domain, *calculated_sliders
            )

            if "Error" in str(status_msg) or "Hata" in str(status_msg):
                return (
                    status_msg,
                    gr.update(),              # Keep pref page visible
                    gr.update(visible=False), # Keep confirm hidden
                    gr.update(),              # confirm prefs html unchanged
                    new_state
                )

            # Preferences saved — show confirmation page with full profile
            profile_path = new_state["user_profiles"][domain]
            value_note = confirm_pages[language]["value_note"]
            pref_html = generate_preferences_html(profile_path, domain, value_note=value_note)

            return (
                status_msg,
                gr.update(visible=False),    # Hide pref page
                gr.update(visible=True),     # Show confirm page
                gr.update(value=pref_html),  # Populate preferences HTML
                new_state
            )

        return handler

    def create_confirm_handler(domain, language):
        confirm_config = confirm_pages[language]

        def handler(state):
            current_idx = state.get("current_negotiation_index", 0)

            try:
                initialize_session(state)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return (
                    gr.update(visible=True),  # Keep confirm page visible (show error via status)
                    gr.update(visible=False),  # Keep video page hidden
                    gr.update(),               # video component unchanged
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                    gr.update(), gr.update(),  # offer_panel_html_1, offer_panel_html_2
                    state
                )

            lang = state["language"]
            negotiation_domain, _ = state["negotiations"][current_idx]
            instructions_html = get_negotiation_instructions(lang, negotiation_domain)

            profile_path = state["user_profiles"][domain]
            pref_html = generate_preferences_html(profile_path, domain)

            pref_updates = [
                gr.update(value=pref_html) if i == current_idx else gr.update()
                for i in range(2)
            ]
            instructions_updates = [
                gr.update(value=instructions_html) if i == current_idx else gr.update()
                for i in range(2)
            ]

            offer_html = generate_offer_panel_html(profile_path, lang, current_idx + 1)
            offer_panel_updates = [
                gr.update(value=offer_html) if i == current_idx else gr.update()
                for i in range(2)
            ]

            video_url = VIDEO_EXPLANATION_PATH_TR if lang == "Türkçe" else VIDEO_EXPLANATION_PATH_EN

            return (
                gr.update(visible=False),                # Hide confirm page
                gr.update(visible=True),                 # Show video page
                gr.update(value=build_video_explanation_embed(video_url)),  # Set video
                gr.update(visible=False),                # Keep negotiation_page_1 hidden
                gr.update(visible=False),                # Keep negotiation_page_2 hidden
                update_progress_text(state),
                *pref_updates,                           # 2 preference display updates
                *instructions_updates,                       # 2 instruction-html updates
                *offer_panel_updates,                    # 2 offer panel HTML updates
                state
            )

        return handler

    # Wire all preference pages
    for (domain, language), pref_config in pref_pages.items():
        # Build list of all slider inputs for this page
        all_sliders = pref_config["issue_sliders"] + [
            slider
            for issue_sliders in pref_config["value_sliders"].values()
            for slider in issue_sliders
        ]
        
        # 1. BIND AUTO-BALANCING
        issue_sliders = pref_config["issue_sliders"]
        sum_disp = pref_config["sum_display"]
        
        # We need a closure to capture the sliders for this specific page
        def make_balance_handler(sliders_list, display_comp):
            def balance_fn(changed_val, changed_idx, *current_values):
                # current_values is a tuple of all slider values
                current_values_list = list(current_values)
                # Normalize
                new_values = normalize_issue_weights_on_change(changed_idx, changed_val, current_values_list)
                # Formatting for display
                total_text = update_sum_display(new_values)
                # Return updates for all sliders + sum display
                return new_values + [total_text]
            return balance_fn

        # Bind each slider to auto-normalize weights to 100%
        for idx, slider in enumerate(issue_sliders):
            def create_change_trigger(s_idx, s_list, d_comp):
                def trigger(val, *others):
                    all_vals = list(others)
                    all_vals[s_idx] = val

                    new_vals = normalize_issue_weights_on_change(s_idx, val, all_vals)
                    total_txt = update_sum_display(new_vals)
                    return new_vals + [total_txt]

                return trigger

            trigger_fn = create_change_trigger(idx, issue_sliders, sum_disp)

            slider.release(
                fn=trigger_fn,
                inputs=[slider] + issue_sliders,
                outputs=issue_sliders + [sum_disp]
            )

        # 2. BIND SUBMIT BUTTON
        # No sliders needed as inputs - calculate everything from order data
        order_textbox = pref_config["order_textbox"]
        issue_container_id = f"issue_sort_{domain}_{language[:2]}"
        
        # JS reads current order from DOM and returns it
        read_order_js = f'''
        function(state, orderText) {{
            // Read current issue order from DOM
            var orderData = {{issues: [], values: {{}}}};
            
            // Read issue container
            var container = document.getElementById("{issue_container_id}");
            if (container) {{
                var items = container.querySelectorAll(".sortable-item");
                for (var i = 0; i < items.length; i++) {{
                    orderData.issues.push(items[i].getAttribute("data-id"));
                }}
            }}
            console.log("Read issue order from DOM:", orderData.issues);
            
            // Read value containers (for holiday domain)
            var valueContainers = document.querySelectorAll("[id^='values_']");
            for (var j = 0; j < valueContainers.length; j++) {{
                var vc = valueContainers[j];
                var parts = vc.id.split('_');
                if (parts.length >= 2) {{
                    var issueName = parts[1];
                    var valueItems = vc.querySelectorAll(".sortable-item");
                    var vals = [];
                    for (var k = 0; k < valueItems.length; k++) {{
                        vals.push(valueItems[k].getAttribute("data-id"));
                    }}
                    if (vals.length > 0) {{
                        orderData.values[issueName] = vals;
                    }}
                }}
            }}
            console.log("Read value orders from DOM:", orderData.values);
            
            return [state, JSON.stringify(orderData)];
        }}
        '''
        
        confirm_config = confirm_pages[language]

        pref_config["btn"].click(
            fn=create_pref_submit_handler(domain, language),
            inputs=[user_state, order_textbox],
            js=read_order_js,
            concurrency_limit=1,
            concurrency_id="preference_submit",
            outputs=[
                pref_config["status"],
                pref_config["page"],
                confirm_config["page"],
                confirm_config["prefs_html"],
                user_state
            ]
        )

        confirm_config["btn"].click(
            fn=create_confirm_handler(domain, language),
            inputs=[user_state],
            outputs=[
                confirm_config["page"],
                video_explanation_page,
                video_component,
                negotiation_page_1,
                negotiation_page_2,
                progress_text_1,
                preferences_display_1,
                preferences_display_2,
                instructions_html_1,
                instructions_html_2,
                offer_panel_html_1,
                offer_panel_html_2,
                user_state
            ]
        )

    # Wire video explanation page — transitions to the correct negotiation page
    def handle_video_continue(state):
        current_idx = state.get("current_negotiation_index", 0)
        page_updates = [gr.update(visible=(i == current_idx)) for i in range(2)]
        return (
            gr.update(visible=False),  # Hide video page
            *page_updates,             # Show correct negotiation page
            state
        )

    video_continue_btn.click(
        fn=handle_video_continue,
        inputs=[user_state],
        outputs=[video_explanation_page, negotiation_page_1, negotiation_page_2, user_state]
    )

    # Continue button on a finished negotiation page → show the per-session
    # survey for the agent the user JUST faced. Session cleanup, index
    # increment, and next-negotiation init happen in the survey-complete
    # handler below, so the user can't skip the survey.
    def handle_negotiation_continue(state):
        current_idx = state.get("current_negotiation_index", 0)
        agent_type = state["negotiations"][current_idx][1]
        if agent_type not in AGENT_SURVEY_URLS:
            raise ValueError(f"Unknown agent_type: {agent_type!r}")
        print(f"Negotiation {current_idx + 1}/2 complete — showing survey for agent_type={agent_type}")
        return (
            gr.update(visible=False),                          # Hide negotiation page 1
            gr.update(visible=False),                          # Hide negotiation page 2
            gr.update(visible=(agent_type == "LLM")),          # LLM survey
            gr.update(visible=(agent_type == "Traditional")),  # Heuristic survey
            state
        )

    for btn in continue_btns:
        btn.click(
            fn=handle_negotiation_continue,
            inputs=[user_state],
            outputs=[
                negotiation_page_1, negotiation_page_2,
                llm_survey_page, heuristic_survey_page,
                user_state
            ]
        )

    # After the per-session survey: clean up the just-finished session, advance
    # the index, and either set up the next negotiation or show thank-you.
    def handle_per_session_survey_complete(state):
        current_session_id = state["session_id"]
        try:
            if current_session_id:
                negotiation_service.cleanup_session(current_session_id)
                print(f"Cleaned up backend session: {current_session_id}")
        except Exception as e:
            print(f"Error cleaning up session {current_session_id}: {e}")

        if current_session_id in user_queues:
            del user_queues[current_session_id]

        state["current_negotiation_index"] += 1
        state["session_started"] = False
        state["negotiation_start_time"] = None

        current_idx = state["current_negotiation_index"]
        print(f"Survey complete after negotiation {current_idx}/2")

        timer_reset = gr.update(value=update_timer_html(None))

        if current_idx < 2:
            initialize_session(state)

            domain, _ = state["negotiations"][current_idx]
            profile_path = state["user_profiles"][domain]
            preferences_html = generate_preferences_html(profile_path, domain)
            language = state["language"]
            instructions_html = get_negotiation_instructions(language, domain)
            offer_html = generate_offer_panel_html(profile_path, language, current_idx + 1)

            page_updates = [gr.update(visible=(i == current_idx)) for i in range(2)]
            progress_update = update_progress_text(state)
            offer_panel_updates = [
                gr.update(value=offer_html) if i == current_idx else gr.update()
                for i in range(2)
            ]

            return (
                *page_updates,                            # 2 negotiation page visibility updates
                gr.update(visible=False),                 # Hide thank you
                gr.update(visible=False),                 # Hide LLM survey
                gr.update(visible=False),                 # Hide Heuristic survey
                progress_update, progress_update,
                gr.update(visible=False), gr.update(visible=False),  # 2 continue btns
                gr.update(visible=False), gr.update(visible=False),  # 2 status texts
                timer_reset, timer_reset,
                gr.update(visible=True), gr.update(visible=True),    # 2 chat rows
                gr.update(value=preferences_html), gr.update(value=preferences_html),
                gr.update(value=instructions_html), gr.update(value=instructions_html),
                *offer_panel_updates,
                state
            )
        else:
            print("All negotiations + surveys complete - showing thank-you")
            return (
                gr.update(visible=False),                 # Hide negotiation page 1
                gr.update(visible=False),                 # Hide negotiation page 2
                gr.update(visible=True),                  # Show thank you
                gr.update(visible=False),                 # Hide LLM survey
                gr.update(visible=False),                 # Hide Heuristic survey
                gr.update(value=""), gr.update(value=""),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
                timer_reset, timer_reset,
                gr.update(visible=True), gr.update(visible=True),
                gr.update(value=""), gr.update(value=""),
                gr.update(value=""), gr.update(value=""),
                gr.update(value=""), gr.update(value=""),
                state
            )

    _per_session_survey_outputs = [
        negotiation_page_1, negotiation_page_2,
        thank_you_page,
        llm_survey_page, heuristic_survey_page,
        progress_text_1, progress_text_2,
        continue_btn_1, continue_btn_2,
        status_text_1, status_text_2,
        timer_display_1, timer_display_2,
        chat_row_1, chat_row_2,
        preferences_display_1, preferences_display_2,
        instructions_html_1, instructions_html_2,
        offer_panel_html_1, offer_panel_html_2,
        user_state
    ]

    for survey_btn in (llm_survey_btn, heuristic_survey_btn):
        survey_btn.click(
            fn=handle_per_session_survey_complete,
            inputs=[user_state],
            outputs=_per_session_survey_outputs
        )

    def handle_accept_action(state: dict, offer_json: str, lock_json: str, history):
        """
        Send an accept message directly from offer panel (bypasses chat textbox submit).
        """
        accept_message = "Kabul ediyorum" if state.get("language") == "Türkçe" else "I accept"
        response, reset_offer_json, locked_offer_json = chat(
            accept_message,
            history or [],
            state,
            offer_json,
            lock_json
        )

        updated_history = list(history or [])
        updated_history.append({"role": "user", "content": accept_message})
        updated_history.append({"role": "assistant", "content": response})

        return updated_history, reset_offer_json, locked_offer_json

    def handle_send_offer_action(state: dict, offer_json: str, lock_json: str, history, message: str):
        """
        Send current textbox message + selected offer via panel Send Offer bridge.
        """
        clean_message = (message or "").strip()
        if not clean_message:
            return list(history or []), offer_json, "", "{}"

        response, reset_offer_json, locked_offer_json = chat(
            clean_message,
            history or [],
            state,
            offer_json,
            lock_json
        )
        updated_history = list(history or [])
        updated_history.append({"role": "user", "content": clean_message})
        updated_history.append({"role": "assistant", "content": response})

        return updated_history, reset_offer_json, "", locked_offer_json

    # Accept bridge wiring for both isolated negotiation pages
    accept_bridge_btn_1.click(
        fn=handle_accept_action,
        inputs=[user_state, offer_textbox_1, offer_lock_tb_1, chat_interface_1.chatbot],
        outputs=[chat_interface_1.chatbot, offer_textbox_1, offer_lock_tb_1]
    )
    accept_bridge_btn_2.click(
        fn=handle_accept_action,
        inputs=[user_state, offer_textbox_2, offer_lock_tb_2, chat_interface_2.chatbot],
        outputs=[chat_interface_2.chatbot, offer_textbox_2, offer_lock_tb_2]
    )
    send_offer_bridge_btn_1.click(
        fn=handle_send_offer_action,
        inputs=[user_state, offer_textbox_1, offer_lock_tb_1, chat_interface_1.chatbot, send_offer_msg_tb_1],
        outputs=[chat_interface_1.chatbot, offer_textbox_1, send_offer_msg_tb_1, offer_lock_tb_1]
    )
    send_offer_bridge_btn_2.click(
        fn=handle_send_offer_action,
        inputs=[user_state, offer_textbox_2, offer_lock_tb_2, chat_interface_2.chatbot, send_offer_msg_tb_2],
        outputs=[chat_interface_2.chatbot, offer_textbox_2, send_offer_msg_tb_2, offer_lock_tb_2]
    )

    # Poll every second to check if current negotiation has completed
    def check_status_for_current_page(state):
        idx = state.get("current_negotiation_index", 0)
        btn_update, status_update, state, timer_update, chat_row_update = check_negotiation_status(state)

        # Route updates to the active page; no-op for the other
        results = []
        for i in range(2):
            if i == idx:
                results.extend([btn_update, status_update, timer_update, chat_row_update])
            else:
                results.extend([gr.update(), gr.update(), gr.update(), gr.update()])
        results.append(state)
        return tuple(results)

    timer = gr.Timer(1)
    timer.tick(
        fn=check_status_for_current_page,
        inputs=[user_state],
        outputs=[
            continue_btn_1, status_text_1, timer_display_1, chat_row_1,
            continue_btn_2, status_text_2, timer_display_2, chat_row_2,
            user_state
        ]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=20)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )
