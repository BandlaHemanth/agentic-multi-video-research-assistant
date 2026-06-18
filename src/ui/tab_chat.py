"""
tab_chat.py — ChatGPT-style Chat Assistant interface with a split layout (SaaS Redesign).

Flow (no trigger_query / double-rerun):
  1. st.chat_input() captures user prompt
  2. Immediately append user message to chat_history
  3. Call agent.run(prompt) in the same execution cycle
  4. Stream the assistant response
  5. Append the assistant response to chat_history
  6. st.rerun() once to lock in the final state

Regeneration uses st.session_state["regen_query"] — the only case that needs
a rerun-to-execute bridge, because button clicks inherently trigger a rerun.
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


def _log(label: str, detail: str = ""):
    """Structured log line for debugging the chat pipeline."""
    msg = f"[CHAT] {label}" + (f" — {detail}" if detail else "")
    logger.info(msg)
    print(msg)


def markdown_to_html(text: str, is_user: bool = False) -> str:
    """
    Converts basic markdown elements (bold, lists, linebreaks, links) to HTML
    suitable for rendering inside custom chat bubbles.
    """
    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)

    link_color = "#ffffff" if is_user else "#7C5CFC"
    html = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        fr'<a href="\2" target="_blank" style="color: {link_color}; font-weight: 600; text-decoration: underline;">\1</a>',
        html
    )

    html = html.replace("\n", "<br>")

    lines = html.split("<br>")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("* ") or stripped.startswith("- "):
            lines[i] = f'<li style="margin-left: 1.5rem; margin-top: 0.25rem;">{stripped[2:]}</li>'
    html = "<br>".join(lines)

    return html


def _execute_query(query: str, agent: VideoResearchAgent, rag_manager: HybridRAGManager):
    """
    Core execution path shared by new prompts and regeneration.
    Runs agent.run(), streams the response, appends to chat_history, then reruns.
    """
    # Guard: empty index
    if not rag_manager.chunks:
        _log("EMPTY INDEX", "No chunks — telling user to index a video first")
        error_msg = "⚠️ The search index is currently empty. Please index a YouTube video or playlist in the sidebar first."
        st.markdown(f'''
        <div class="chat-message-row assistant-row">
            <div style="margin-right: 0.5rem; font-size: 1.25rem;">🤖</div>
            <div class="chat-bubble assistant-bubble">{error_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        return

    _log("agent.run() ENTER", f"query='{query[:80]}', chunks={len(rag_manager.chunks)}")
    run_start = time.time()

    with st.spinner("Assistant is searching transcripts and reasoning..."):
        try:
            result: AgentExecutionResult = agent.run(query)
            elapsed = time.time() - run_start
            _log("agent.run() RETURNED", f"elapsed={elapsed:.2f}s, answer_len={len(result.answer)}")

            clean_answer = result.answer

            # Stream the response word-by-word into a chat bubble
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

            # Persist assistant message with trace metadata
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": clean_answer,
                "trace_steps": [ts.__dict__ for ts in result.trace_steps],
                "retrieved_chunks": [c.__dict__ for c in result.retrieved_chunks]
            })
            _log("HISTORY APPENDED", f"history_len={len(st.session_state.chat_history)}")

            # Single rerun to lock in the final state
            _log("st.rerun()", "locking in chat history")
            st.rerun()

        except Exception as e:
            full_tb = tb_module.format_exc()
            elapsed = time.time() - run_start
            _log("EXCEPTION", f"elapsed={elapsed:.2f}s\n{full_tb}")
            logger.critical(f"[CRITICAL] Chat execution failed:\n{full_tb}")
            st.error("❌ A critical error occurred in the chat assistant:")
            st.exception(e)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"An error occurred: {e}"
            })


def render_chat_tab(agent: VideoResearchAgent, rag_manager: HybridRAGManager, api_key: str):
    """
    Renders the Chat Assistant interface.
    """
    # 1. Fallback Mode banner
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

    # 2. Layout — split in Debug Mode
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

        # Clear Chat button
        col_header_left, col_header_right = st.columns([5, 1])
        with col_header_right:
            if st.button("Clear Chat", key="chat_clear_btn", help="Reset conversation"):
                st.session_state.chat_history = []
                st.rerun()

        # ─────────────────────────────────────────────────────────────
        # 3. Render conversation history
        # ─────────────────────────────────────────────────────────────
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

            # Inline action buttons under assistant messages
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
                            # Trim history to remove this assistant answer
                            st.session_state.chat_history = st.session_state.chat_history[:idx]
                            # Store the query for re-execution after rerun
                            st.session_state["regen_query"] = last_query
                            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────
        # 4. Handle pending regeneration (from button click above)
        # ─────────────────────────────────────────────────────────────
        regen_query = st.session_state.pop("regen_query", None)
        if regen_query:
            _log("REGEN", f"Re-executing query: '{regen_query[:80]}'")
            _execute_query(regen_query, agent, rag_manager)
            # _execute_query calls st.rerun() on success, so we won't reach here
            return

        # ─────────────────────────────────────────────────────────────
        # 5. Handle new user input — no trigger_query, no extra rerun
        # ─────────────────────────────────────────────────────────────
        prompt = st.chat_input("Ask about the indexed videos...")

        if prompt:
            _log("NEW PROMPT", f"'{prompt[:80]}'")

            # Immediately append user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Execute in the same cycle
            _execute_query(prompt, agent, rag_manager)
            # _execute_query calls st.rerun() on success, so we won't reach here

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
