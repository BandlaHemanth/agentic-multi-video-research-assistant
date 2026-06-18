"""
tab_chat.py — ChatGPT-style Chat Assistant interface with a split layout (SaaS Redesign).
Contains aligned right/left chat bubbles, typewriter response streaming, inline actions (like/dislike/copy/regen),
and a dedicated right-hand Execution Trace Panel in Debug Mode.
"""

import re
import time
import traceback as tb_module
import streamlit as st
import logging
from typing import List, Dict, Any

from src.core.agent import VideoResearchAgent, AgentExecutionResult
from src.core.rag import HybridRAGManager, format_time
from src.ui.utils import parse_citations, stream_text, is_demo_fallback_active, render_trace_panel

logger = logging.getLogger(__name__)


def _checkpoint(label: str, passed: bool, detail: str = ""):
    """Emit a PASS/FAIL checkpoint to both the Python logger and stdout."""
    status = "PASS" if passed else "FAIL"
    msg = f"[CHECKPOINT] {status} — {label}" + (f": {detail}" if detail else "")
    if passed:
        logger.info(msg)
    else:
        logger.error(msg)
    print(msg)


def markdown_to_html(text: str, is_user: bool = False) -> str:
    """
    Converts basic markdown elements (bold, lists, linebreaks, links) to HTML
    suitable for rendering inside custom chat bubbles.
    """
    # 1. Escaping basic HTML to prevent injection, but keeping brackets for links
    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2. Convert bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)

    # 3. Convert links: [text](url) -> anchor tag
    link_color = "#ffffff" if is_user else "#7C5CFC"
    html = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        fr'<a href="\2" target="_blank" style="color: {link_color}; font-weight: 600; text-decoration: underline;">\1</a>',
        html
    )

    # 4. Linebreaks
    html = html.replace("\n", "<br>")

    # 5. Lists (converts lines starting with * or - into list elements)
    lines = html.split("<br>")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("* ") or stripped.startswith("- "):
            lines[i] = f'<li style="margin-left: 1.5rem; margin-top: 0.25rem;">{stripped[2:]}</li>'
    html = "<br>".join(lines)

    return html


def render_chat_tab(agent: VideoResearchAgent, rag_manager: HybridRAGManager, api_key: str):
    """
    Renders the Chat Assistant interface.
    """
    # 1. Fallback Mode banner check
    last_answer = ""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        assistant_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
        if assistant_msgs:
            last_answer = assistant_msgs[-1]["content"]

    if is_demo_fallback_active(api_key, last_answer):
        st.markdown(
            '<div class="fallback-banner">'
            '  <span>⚠️ <strong>Gemini API unavailable. Running in DEMO FALLBACK mode.</strong></span>'
            '</div>',
            unsafe_allow_html=True
        )

    # Main structure: Split layout in Debug Mode, full-width otherwise
    debug_active = st.session_state.get("debug_mode", False)

    if debug_active:
        col_chat, col_trace = st.columns([5, 3], gap="large")
    else:
        col_chat = st.container()
        col_trace = None

    with col_chat:
        st.subheader("💬 Chat Assistant")
        st.markdown("*Search, summarize, compare and analyze video transcripts with conversational AI.*")

        # Initialize history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Clear Chat Button Row
        col_header_left, col_header_right = st.columns([5, 1])
        with col_header_right:
            if st.button("Clear Chat", key="chat_clear_btn", help="Reset conversation"):
                st.session_state.chat_history = []
                st.rerun()

        # Render conversation history inside custom CSS bubbles
        st.markdown('<div class="chat-bubble-container">', unsafe_allow_html=True)
        for idx, msg in enumerate(st.session_state.chat_history):
            role = msg["role"]
            content = msg["content"]
            is_user = (role == "user")

            citations_parsed = parse_citations(content, rag_manager.video_metadata_map)
            html_content = markdown_to_html(citations_parsed, is_user=is_user)

            row_class = "user-row" if is_user else "assistant-row"
            bubble_class = "user-bubble" if is_user else "assistant-bubble"
            avatar = "👤" if is_user else "🤖"

            st.markdown(f'''
            <div class="chat-message-row {row_class}">
                <div style="margin-right: 0.5rem; font-size: 1.25rem;">{avatar if not is_user else ""}</div>
                <div class="chat-bubble {bubble_class}">
                    {html_content}
                </div>
                <div style="margin-left: 0.5rem; font-size: 1.25rem;">{avatar if is_user else ""}</div>
            </div>
            ''', unsafe_allow_html=True)

            if not is_user:
                col_btn1, col_btn2, col_btn3, col_btn4, _ = st.columns([1, 1, 1, 1, 15])
                with col_btn1:
                    if st.button("📋", key=f"copy_{idx}", help="Copy response text"):
                        st.toast("Copied to clipboard!")
                        st.session_state["copied_text"] = content
                with col_btn2:
                    if st.button("👍", key=f"like_{idx}", help="Thumbs up"):
                        st.toast("Thanks for the feedback!")
                with col_btn3:
                    if st.button("👎", key=f"dislike_{idx}", help="Thumbs down"):
                        st.toast("Feedback recorded.")
                with col_btn4:
                    if st.button("🔄", key=f"regen_{idx}", help="Regenerate this response"):
                        user_queries = [m for m in st.session_state.chat_history[:idx] if m["role"] == "user"]
                        if user_queries:
                            last_query = user_queries[-1]["content"]
                            st.session_state.chat_history = st.session_state.chat_history[:idx]
                            st.session_state["trigger_query"] = last_query
                            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ── CHECKPOINT A: Capture chat input ──────────────────────────────
        triggered_query = st.session_state.pop("trigger_query", None)
        user_input = st.chat_input("Ask about the indexed videos...")

        _checkpoint("A — chat_input captured", True,
                    f"user_input={'<received>' if user_input else '<empty>'}, "
                    f"trigger_query={'<present>' if triggered_query else '<empty>'}")

        active_input = triggered_query or user_input

        if active_input:
            if not triggered_query:
                # ── CHECKPOINT B: Storing trigger_query ───────────────────
                _checkpoint("B — New query received, storing trigger_query and calling st.rerun()", True,
                            f"query='{active_input[:60]}'")
                st.session_state.chat_history.append({"role": "user", "content": active_input})
                st.session_state["trigger_query"] = active_input
                st.rerun()

            else:
                # ── CHECKPOINT C: trigger_query restored after rerun ──────
                _checkpoint("C — trigger_query restored after rerun", True,
                            f"query='{active_input[:60]}'")

                # Render User bubble for this turn
                st.markdown(f'''
                <div class="chat-message-row user-row">
                    <div class="chat-bubble user-bubble">
                        {markdown_to_html(active_input, is_user=True)}
                    </div>
                    <div style="margin-left: 0.5rem; font-size: 1.25rem;">👤</div>
                </div>
                ''', unsafe_allow_html=True)

                # ── CHECKPOINT D: Empty database guard ────────────────────
                if not rag_manager.chunks:
                    _checkpoint("D — rag_manager.chunks check", False, "Index is empty")
                    error_msg = "⚠️ The search index is currently empty. Please index a YouTube video or playlist in the sidebar first."
                    st.markdown(f'''
                    <div class="chat-message-row assistant-row">
                        <div style="margin-right: 0.5rem; font-size: 1.25rem;">🤖</div>
                        <div class="chat-bubble assistant-bubble">{error_msg}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    return

                _checkpoint("D — rag_manager.chunks check", True, f"{len(rag_manager.chunks)} chunks available")

                # ── CHECKPOINT E: agent.run() entry ──────────────────────
                _checkpoint("E — agent.run() ENTERING", True, f"query='{active_input[:60]}'")
                run_start = time.time()

                with st.spinner("Assistant is searching transcripts and reasoning..."):
                    try:
                        result: AgentExecutionResult = agent.run(active_input)

                        run_elapsed = time.time() - run_start

                        # ── CHECKPOINT F: agent.run() returned ───────────
                        _checkpoint("F — agent.run() RETURNED", True,
                                    f"elapsed={run_elapsed:.2f}s, answer_len={len(result.answer)}")

                        clean_answer = result.answer

                        # Streaming display container
                        response_container = st.empty()
                        response_text = ""

                        for word in stream_text(clean_answer):
                            response_text += word
                            citations_parsed = parse_citations(response_text, rag_manager.video_metadata_map)
                            html_text = markdown_to_html(citations_parsed, is_user=False)

                            response_container.markdown(f'''
                            <div class="chat-message-row assistant-row">
                                <div style="margin-right: 0.5rem; font-size: 1.25rem;">🤖</div>
                                <div class="chat-bubble assistant-bubble">
                                    {html_text}
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)

                        # ── CHECKPOINT G: append to chat_history ─────────
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": clean_answer,
                            "trace_steps": [ts.__dict__ for ts in result.trace_steps],
                            "retrieved_chunks": [c.__dict__ for c in result.retrieved_chunks]
                        })
                        _checkpoint("G — assistant response appended to st.session_state.chat_history", True,
                                    f"history_len={len(st.session_state.chat_history)}")

                        # ── CHECKPOINT H: st.rerun() ──────────────────────
                        _checkpoint("H — st.rerun() EXECUTING", True, "")
                        st.rerun()

                    except Exception as e:
                        full_tb = tb_module.format_exc()
                        run_elapsed = time.time() - run_start
                        _checkpoint("E/F — agent.run() RAISED EXCEPTION", False,
                                    f"elapsed={run_elapsed:.2f}s\n{full_tb}")
                        logger.critical(f"[CRITICAL] Chat execution failed:\n{full_tb}")
                        print(f"[CRITICAL] Chat execution failed:\n{full_tb}")
                        st.error("❌ A critical error occurred in the chat assistant:")
                        st.exception(e)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"An error occurred: {e}"
                        })

    # ─────────────────────────────────────────────────────────────────────
    # RIGHT COLUMN: EXECUTION TRACE PANEL (DEBUG MODE ONLY)
    # ─────────────────────────────────────────────────────────────────────
    if debug_active and col_trace is not None:
        with col_trace:
            trace_steps = []
            retrieved_chunks = []
            if "chat_history" in st.session_state and st.session_state.chat_history:
                assistant_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
                if assistant_msgs:
                    last_msg = assistant_msgs[-1]
                    trace_steps = last_msg.get("trace_steps", [])
                    retrieved_chunks = last_msg.get("retrieved_chunks", [])

            render_trace_panel(trace_steps, retrieved_chunks, rag_manager)
