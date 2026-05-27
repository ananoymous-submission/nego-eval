import os
import time
import gradio as gr

def create_timer_display():
    """
    Create the initial timer display component.
    """
    # Read deadline from environment variable (in seconds)
    deadline_seconds = int(os.getenv("NEGOTIATION_DEADLINE_TIME", 900))
    minutes = deadline_seconds // 60
    seconds = deadline_seconds % 60

    return gr.HTML(f"""
        <div style="text-align: center; margin: 20px 0; min-height: 120px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-size: 2.5em; color: #16a34a; font-family: monospace; font-weight: bold; line-height: 1.2;">
                {minutes}:{seconds:02d}
            </div>
            <div style="color: #6b7280; margin-top: 10px; font-size: 1rem; line-height: 1.5;">Time Remaining / Kalan Süre</div>
        </div>
    """, visible=True)

def update_timer_html(start_time):
    """
    Generate the updated HTML for the timer based on start time.

    Args:
        start_time: Timestamp when negotiation started, or None.

    Returns:
        HTML string for the timer.
    """
    # Read deadline from environment variable (in seconds)
    total_seconds = int(os.getenv("NEGOTIATION_DEADLINE_TIME", 900))

    if not start_time:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        # Default Green
        color = "#16a34a"
    else:
        elapsed = time.time() - start_time
        remaining = max(0, total_seconds - elapsed)

        minutes = int(remaining // 60)
        seconds = int(remaining % 60)

        # Color logic
        # Last minute: Red
        # Last third of time: Orange
        # Otherwise: Green
        if remaining <= 60:
            color = "#dc2626" # Red
        elif remaining <= (total_seconds / 3):
            color = "#ea580c" # Orange
        else:
            color = "#16a34a" # Green

    return f"""
        <div style="text-align: center; margin: 20px 0; min-height: 120px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-size: 2.5em; color: {color}; font-family: monospace; font-weight: bold; line-height: 1.2;">
                {minutes}:{seconds:02d}
            </div>
            <div style="color: #6b7280; margin-top: 10px; font-size: 1rem; line-height: 1.5;">Time Remaining / Kalan Süre</div>
        </div>
    """
