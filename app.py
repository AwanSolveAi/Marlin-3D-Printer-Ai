from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================
# PROJECT PATHS / IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from answer_generator import MarlinAnswerAgent  # noqa: E402


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TechSolAi Marlin Support AI",
    page_icon="🛠️",
    layout="wide",
)


# ============================================================
# STYLES
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .techsolai-header {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            margin-bottom: 1rem;
        }

        .techsolai-subtitle {
            opacity: 0.78;
            margin-top: 0.2rem;
        }

        .meta-box {
            padding: 0.55rem 0.75rem;
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 10px;
            margin-bottom: 0.75rem;
        }

        .question-box {
            padding: 0.7rem 0.85rem;
            border-left: 4px solid rgba(128,128,128,0.45);
            background: rgba(128,128,128,0.06);
            border-radius: 8px;
            margin-bottom: 0.85rem;
        }

        .history-card {
            padding: 0.9rem 1rem;
            border: 1px solid rgba(128,128,128,0.20);
            border-radius: 12px;
            margin-bottom: 1rem;
        }

        .small-note {
            font-size: 0.90rem;
            opacity: 0.78;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# ============================================================
# AGENT
# ============================================================

@st.cache_resource(show_spinner="Loading Marlin documentation and retrieval index...")
def get_agent() -> MarlinAnswerAgent:
    return MarlinAnswerAgent()


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def get_route(result: dict[str, Any]) -> str:
    return safe_text(
        result.get("query_type")
        or result.get("route")
        or result.get("intent")
        or "unknown"
    )


def normalize_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("sources") or []
    normalized: list[dict[str, Any]] = []

    for i, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            normalized.append(
                {
                    "rank": item.get("rank", i),
                    "document": item.get("document", ""),
                    "section": item.get("section", ""),
                    "source": item.get("source", ""),
                    "chunk_id": item.get("chunk_id", ""),
                }
            )
        else:
            normalized.append(
                {
                    "rank": i,
                    "document": "",
                    "section": "",
                    "source": safe_text(item),
                    "chunk_id": "",
                }
            )

    return normalized


def run_question(question: str) -> dict[str, Any]:
    agent = get_agent()
    result = agent.generate(question)

    if isinstance(result, dict):
        answer = safe_text(result.get("answer"), "No answer returned.")
        route = get_route(result)
        sources = normalize_sources(result)
    else:
        answer = safe_text(result, "No answer returned.")
        route = "unknown"
        sources = []

    return {
        "question": question,
        "answer": answer,
        "route": route,
        "sources": sources,
    }


def render_result(item: dict[str, Any], latest: bool = False) -> None:
    if latest:
        st.subheader("Latest Answer")
    else:
        st.markdown("---")

    st.markdown(
        f'<div class="question-box"><strong>Question:</strong><br>{safe_text(item.get("question"))}</div>',
        unsafe_allow_html=True,
    )

    route = safe_text(item.get("route"), "unknown")
    st.markdown(
        f'<div class="meta-box"><strong>Route / Query Type:</strong> {route}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Answer**")
    st.markdown(safe_text(item.get("answer"), "No answer returned."))

    sources = item.get("sources") or []

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        if not sources:
            st.caption("No supporting sources were returned for this response.")
        else:
            for source in sources:
                rank = source.get("rank", "")
                document = safe_text(source.get("document"))
                section = safe_text(source.get("section"))
                source_path = safe_text(source.get("source"))
                chunk_id = safe_text(source.get("chunk_id"))

                title = document or source_path or f"Source {rank}"
                st.markdown(f"**{rank}. {title}**")

                if section:
                    st.write(f"Section: {section}")
                if source_path:
                    st.caption(f"Source: {source_path}")
                if chunk_id:
                    st.caption(f"Chunk ID: {chunk_id}")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="techsolai-header">
        <h1 style="margin-bottom:0.2rem;">TechSolAi — Marlin Support AI</h1>
        <div class="techsolai-subtitle">
            Documentation-grounded technical support demo for Marlin 3D printer firmware.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Demo scope: Answers are generated from the indexed Marlin documentation. "
    "Unsupported or insufficiently grounded requests may be declined. "
    "This demo does not replace manufacturer safety procedures or qualified technical judgment."
)


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask a Marlin Question")

st.markdown("**Try an example**")

example_questions = [
    "What does M112 do?",
    "My probe triggers but Z does not move down. What should I check?",
    "The part fan blows toward the heater and I see thermal-runaway errors. Is that cause documented?",
    "My supported Trinamic driver powered on late. What diagnostic command can reinitialize it?",
    "My firmware build is too large. What starting configuration does the troubleshooting guide recommend?",
    "The controller is stuck waiting for temperature. How does M108 differ from a full stop?",
]

example_cols = st.columns(2)
for idx, example in enumerate(example_questions):
    with example_cols[idx % 2]:
        if st.button(
            example,
            key=f"example_{idx}",
            use_container_width=True,
        ):
            st.session_state.question_input = example
            st.rerun()

question = st.text_area(
    "Question",
    key="question_input",
    placeholder="Ask a Marlin troubleshooting, command, or configuration question...",
    height=110,
)

left, right = st.columns([1, 1])

with left:
    ask_clicked = st.button(
        "Ask",
        type="primary",
        use_container_width=True,
    )

with right:
    clear_clicked = st.button(
        "Clear History",
        use_container_width=True,
    )

if clear_clicked:
    st.session_state.history = []
    st.rerun()

if ask_clicked:
    clean_question = question.strip()

    if not clean_question:
        st.warning("Enter a question before clicking Ask.")
    else:
        try:
            with st.spinner("Searching Marlin documentation and preparing answer..."):
                item = run_question(clean_question)

            # Latest question always stays at the top.
            st.session_state.history.insert(0, item)

            # Clear the text box for the next question.
            st.session_state.question_input = ""
            st.rerun()

        except Exception as exc:
            st.error(
                "The demo could not process this question. "
                "Please check that the Marlin index and local model are available."
            )
            with st.expander("Technical details"):
                st.code(repr(exc))


# ============================================================
# LATEST ANSWER + HISTORY
# ============================================================

if st.session_state.history:
    latest = st.session_state.history[0]
    render_result(latest, latest=True)

    if len(st.session_state.history) > 1:
        st.subheader("History")

        for index, item in enumerate(st.session_state.history[1:], start=2):
            with st.expander(
                f"{index - 1}. {safe_text(item.get('question'))[:120]}",
                expanded=False,
            ):
                render_result(item, latest=False)
else:
    st.caption(
        "No questions asked yet. Your latest answer will appear here, "
        "and earlier questions will remain available in History."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div class="small-note">
        <strong>TechSolAi Demo Candidate</strong><br>
        Documentation-grounded technical knowledge assistant.<br>
        Cloud-hosted demo. No API keys or credentials are displayed in this interface.
    </div>
    """,
    unsafe_allow_html=True,
)
