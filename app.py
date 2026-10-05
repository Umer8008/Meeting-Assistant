"""
Meeting Intelligence — Streamlit Application Entry Point.

This is the main entry point: `streamlit run app.py`

Responsibilities:
  - Page configuration (title, icon, layout)
  - CSS injection via gui.styles
  - Session state initialization
  - Page routing based on sidebar navigation
  - Environment variable loading

This file contains ZERO business logic — all AI/ML operations
are delegated to the core/ and utils/ packages.
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env) before any module that needs API keys
load_dotenv()

from gui.styles import inject_css
from gui.components import render_header, render_card, render_metric_row, render_empty_state
from gui.sidebar import render_sidebar
from gui.input_view import render_input_view
from gui.transcript_view import render_transcript_view
from gui.analysis_view import render_analysis_view
from gui.translation_view import render_translation_view
from gui.rag_view import render_rag_view


# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initializes session state with default values on first run.

    Only sets keys that don't already exist — subsequent reruns
    preserve the user's data. This prevents accidental resets.
    """
    defaults = {
        "audio_chunks": None,
        "source_name": None,
        "transcript": None,
        "detected_language": None,
        "mode_label": None,
        "summary": None,
        "title": None,
        "key_points": None,
        "decisions": None,
        "questions": None,
        "action_items": None,
        "vector_store": None,
        "translated_transcript": None,
        "translation_language": None,
        "rag_history": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_home():
    """Renders the home/landing page with a workflow overview."""
    render_header()

    st.markdown(
        '<p style="text-align:center; color:#9ba1b7; font-size:0.95rem; '
        'margin-bottom:2rem;">'
        "Transform your meetings into actionable intelligence. "
        "Upload a recording, get a transcript, and extract insights.</p>",
        unsafe_allow_html=True,
    )

    # ── Workflow steps ──
    col1, col2, col3 = st.columns(3)

    with col1:
        render_card(
            title="1. Upload Media",
            content="Paste a YouTube URL or upload a local audio/video file. "
                    "We'll extract and chunk the audio automatically.",
            icon="📥",
            subtitle="YouTube, MP4, MP3, WAV, and more",
        )
    with col2:
        render_card(
            title="2. Transcribe",
            content="Use Whisper AI to transcribe audio in its original language, "
                    "or translate to English or Urdu in real-time.",
            icon="🎤",
            subtitle="Multilingual • GPU-accelerated",
        )
    with col3:
        render_card(
            title="3. Analyze & Query",
            content="Generate summaries, extract key points, decisions, and action items. "
                    "Ask questions using RAG-powered Q&A.",
            icon="📊",
            subtitle="Title • Summary • Decisions • Q&A",
        )

    st.markdown("---")

    # ── Current session status ──
    has_chunks = bool(st.session_state.get("audio_chunks"))
    has_transcript = bool(st.session_state.get("transcript"))
    has_summary = bool(st.session_state.get("summary"))
    has_vs = bool(st.session_state.get("vector_store"))

    render_metric_row([
        {
            "label": "Media",
            "value": "✅ Loaded" if has_chunks else "—",
            "icon": "📥" if has_chunks else "",
        },
        {
            "label": "Transcript",
            "value": "✅ Ready" if has_transcript else "—",
            "icon": "🎤" if has_transcript else "",
        },
        {
            "label": "Analysis",
            "value": "✅ Done" if has_summary else "—",
            "icon": "📊" if has_summary else "",
        },
        {
            "label": "RAG Q&A",
            "value": "✅ Ready" if has_vs else "—",
            "icon": "💬" if has_vs else "",
        },
    ])

    if not has_chunks:
        st.markdown("")
        render_empty_state(
            "🚀",
            "Ready to get started?",
            "Navigate to 'Input' in the sidebar to upload your first meeting recording.",
        )


def render_settings():
    """Renders the settings page for configuration management."""
    from gui.components import render_section_header, render_alert
    import os

    render_section_header("⚙️", "Settings")

    # ── API Keys Status ──
    render_card(
        title="API Configuration",
        content="Status of required API keys and models.",
        icon="🔑",
    )

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    whisper_model = os.getenv("WHISPER_MODEL", "small")

    col1, col2, col3 = st.columns(3)
    with col1:
        key_status = "✅ Configured" if gemini_key else "❌ Missing"
        st.markdown(
            f'<div class="mi-metric">'
            f'<p class="mi-metric-value">{key_status}</p>'
            f'<p class="mi-metric-label">Gemini API Key</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="mi-metric">'
            f'<p class="mi-metric-value">{gemini_model}</p>'
            f'<p class="mi-metric-label">Gemini Model</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="mi-metric">'
            f'<p class="mi-metric-value">{whisper_model}</p>'
            f'<p class="mi-metric-label">Whisper Model</p></div>',
            unsafe_allow_html=True,
        )

    if not gemini_key:
        render_alert(
            "GEMINI_API_KEY is not set. Summarization, extraction, and RAG features "
            "will not work. Add it to your .env file and restart the app.",
            "error",
        )

    st.markdown("---")

    # ── Session Management ──
    render_card(
        title="Session Management",
        content="Clear cached data and reset the application state.",
        icon="🗑️",
    )

    if st.button("🗑️  Clear All Session Data", key="btn_clear_session"):
        for key in list(st.session_state.keys()):
            if key != "nav_page":  # Preserve navigation selection
                del st.session_state[key]
        init_session_state()
        render_alert("Session data cleared.", "success")
        st.rerun()

    st.markdown("---")

    # ── About ──
    render_card(
        title="About",
        content=(
            "<strong>Meeting Intelligence</strong> v1.0<br>"
            "AI-powered meeting assistant for transcription, analysis, and Q&A.<br><br>"
            "<em>Backend:</em> Whisper (transcription) • Gemini (LLM) • ChromaDB (RAG)<br>"
            "<em>Frontend:</em> Streamlit with custom CSS design system"
        ),
        icon="ℹ️",
    )


# ── Page Router ───────────────────────────────────────────────────
# Maps sidebar selection labels to their corresponding render functions

PAGE_ROUTER = {
    "🏠  Home": render_home,
    "📥  Input": render_input_view,
    "🎤  Transcription": render_transcript_view,
    "📊  Analysis": render_analysis_view,
    "🌐  Translation": render_translation_view,
    "💬  RAG Q&A": render_rag_view,
    "⚙️  Settings": render_settings,
}


def main():
    """Application main function — orchestrates layout and routing."""
    # 1. Inject global CSS design system
    inject_css()

    # 2. Initialize session state defaults
    init_session_state()

    # 3. Render sidebar and get the selected page
    selected_page = render_sidebar()

    # 4. Route to the selected page's render function
    page_fn = PAGE_ROUTER.get(selected_page, render_home)
    page_fn()


if __name__ == "__main__":
    main()
