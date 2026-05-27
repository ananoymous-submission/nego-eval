"""
Video explanation page shown after preference confirmation, before the first negotiation.
Displays a language-dependent instructional YouTube video and a continue button.
The continue button is hidden until the video has been played to the end.
"""

import html
import gradio as gr
from typing import Tuple
from urllib.parse import parse_qs, urlparse


VIDEO_EXPLANATION_COMPONENT_ID = "video-explanation-component"
VIDEO_EXPLANATION_ROOT_ID = "video-explanation-root"
VIDEO_EXPLANATION_PLAYER_ID = "video-explanation-player"
VIDEO_EXPLANATION_CONTINUE_BUTTON_ID = "video-explanation-continue-btn"
VIDEO_EXPLANATION_INIT_GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


def _extract_youtube_video_id(video_url: str) -> str:
    """Extract a YouTube video id from a supported YouTube URL."""
    parsed = urlparse(video_url)
    hostname = (parsed.hostname or "").lower()

    if hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
    elif hostname.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        else:
            video_id = ""
    else:
        video_id = ""

    if not video_id:
        raise ValueError(f"Invalid YouTube URL for video explanation: {video_url}")

    return video_id


def build_video_explanation_embed(video_url: str) -> str:
    """Build the YouTube embed HTML for the explanation page."""
    video_id = _extract_youtube_video_id(video_url)
    embed_url = f"https://www.youtube.com/embed/{video_id}?enablejsapi=1&rel=0"

    return f"""
<div id="{VIDEO_EXPLANATION_ROOT_ID}">
  <div style="position: relative; width: 100%; padding-top: 56.25%; overflow: hidden; border-radius: 12px;">
    <iframe
      id="{VIDEO_EXPLANATION_PLAYER_ID}"
      src="{html.escape(embed_url, quote=True)}"
      title="Negotiation system explanation video"
      style="position: absolute; inset: 0; width: 100%; height: 100%; border: 0;"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      referrerpolicy="strict-origin-when-cross-origin"
      allowfullscreen
    ></iframe>
  </div>
  <img
    src="{VIDEO_EXPLANATION_INIT_GIF}"
    alt=""
    style="display: none;"
    onload="if(window.initVideoExplanation)window.initVideoExplanation()"
  />
</div>
"""


def create_video_explanation_page(video_url_en: str, video_url_tr: str) -> Tuple[gr.Column, gr.components.Component, gr.Button]:
    """Create the video explanation page.

    Args:
        video_url_en: English YouTube URL for the explanation video.
        video_url_tr: Turkish YouTube URL for the explanation video.

    Returns:
        Tuple of (page column, video component, continue button).
    """
    build_video_explanation_embed(video_url_en)
    build_video_explanation_embed(video_url_tr)

    with gr.Column(visible=False) as video_page:
        gr.Markdown("# How to use system / Müzakere sistemi kullanımı videosu")
        gr.Markdown("Please watch until the end. / Lütfen sonuna kadar izleyin.")
        video_component = gr.HTML(
            value=build_video_explanation_embed(video_url_en),
            label="",
            elem_id=VIDEO_EXPLANATION_COMPONENT_ID,
        )
        continue_btn = gr.Button(
            "Continue / Devam Et",
            variant="primary",
            size="lg",
            interactive=False,
            elem_id=VIDEO_EXPLANATION_CONTINUE_BUTTON_ID,
        )

    return video_page, video_component, continue_btn
