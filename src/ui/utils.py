"""
utils.py — Helper utility functions for Streamlit UI component rendering.
Includes text streaming wrappers, citation parsing to clickable timestamped links,
and session state status helper checks.
"""

import re
import time
import logging
from typing import List, Dict, Generator, Optional, Any
from src.core.rag import format_time

logger = logging.getLogger(__name__)

def parse_timestamp_to_seconds(timestamp_str: str) -> int:
    """
    Converts a timestamp string (e.g. MM:SS or H:MM:SS) into total seconds.
    """
    parts = timestamp_str.strip().split(':')
    try:
        if len(parts) == 2:
            # MM:SS
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) >= 3:
            # H:MM:SS or HH:MM:SS
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        pass
    return 0

def resolve_video_id_by_title(title_query: str, video_metadata_map: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Looks up a video ID in the metadata map by performing an exact or partial
    case-insensitive match on the video title.
    """
    cleaned_query = title_query.strip().lower()
    
    # Try exact match
    for v_id, meta in video_metadata_map.items():
        if meta.get("title", "").strip().lower() == cleaned_query:
            return v_id
            
    # Try partial match (query is in video title)
    for v_id, meta in video_metadata_map.items():
        v_title = meta.get("title", "").lower()
        if cleaned_query in v_title or v_title in cleaned_query:
            return v_id
            
    # If no match, try cleaning punctuation
    def clean_str(s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        
    cleaned_query_alphanum = clean_str(cleaned_query)
    for v_id, meta in video_metadata_map.items():
        if clean_str(meta.get("title", "")) == cleaned_query_alphanum:
            return v_id
            
    return None

def parse_citations(text: str, video_metadata_map: Dict[str, Dict[str, Any]]) -> str:
    """
    Searches for citations of format [Video Title - Timestamp] in the text
    and converts them to: [▶ Video Title (Timestamp)](https://youtu.be/video_id?t=seconds).
    
    If the video is not found in the database, falls back to a search on YouTube or a default URL.
    """
    # Pattern to match [Video Title - MM:SS] or [Video Title - H:MM:SS]
    pattern = r"\[([^\]\-]+)\s*-\s*([0-9:]+)\]"
    
    def replace_citation(match: re.Match) -> str:
        title_part = match.group(1).strip()
        timestamp_part = match.group(2).strip()
        
        seconds = parse_timestamp_to_seconds(timestamp_part)
        video_id = resolve_video_id_by_title(title_part, video_metadata_map)
        
        if video_id:
            youtube_url = f"https://youtu.be/{video_id}?t={seconds}"
            # Rendered format: ▶ Video Title (MM:SS) linked directly to the YouTube timestamp
            return f"[▶ {title_part} ({timestamp_part})]({youtube_url})"
        else:
            # Fallback if video ID not resolved
            youtube_url = f"https://www.youtube.com/results?search_query={re.sub(r'[^a-zA-Z0-9 ]', '', title_part).replace(' ', '+')}"
            return f"[▶ {title_part} ({timestamp_part})]({youtube_url})"
            
    return re.sub(pattern, replace_citation, text)

def stream_text(text: str, delay_per_word: float = 0.02) -> Generator[str, None, None]:
    """
    Splits text into words and yields them sequentially with a small delay
    to simulate ChatGPT-style typing/streaming interface in Streamlit.
    """
    if not text:
        return
        
    words = text.split(" ")
    for idx, word in enumerate(words):
        if idx < len(words) - 1:
            yield word + " "
        else:
            yield word
        time.sleep(delay_per_word)

def is_demo_fallback_active(api_key: str, last_response: str = "") -> bool:
    """
    Helper checking whether fallback/simulation mode is active.
    It is active if:
    1. The API key is missing.
    2. The last response contains the '[DEMO FALLBACK]', '[DEMO MODE]', or rate-limit fallback indicators.
    """
    if not api_key.strip():
        return True
    if "[DEMO FALLBACK]" in last_response or "[DEMO MODE]" in last_response or "Gemini API is temporarily unavailable" in last_response:
        return True
    return False

def render_trace_panel(trace_steps: List[Dict[str, Any]], retrieved_chunks: List[Dict[str, Any]], rag_manager: Any):
    """
    Renders the visual execution trace panel with the reasoning steps and RAG scores.
    """
    import streamlit as st
    st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.subheader("🔍 Execution Trace")
    st.markdown("*Real-time pipeline flow, ReAct thoughts, retrieval metrics, and latency.*")
    st.markdown("---")
    
    if not trace_steps:
        st.info("No query executed in this session yet. Run a query to inspect the trace.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    # Render Flowchart
    st.markdown("#### 🔄 Pipeline Operations")
    st.markdown(
        '<div class="trace-flowchart-container">'
        '  <div class="flowchart-step">User Query</div>'
        '  <div class="flowchart-arrow">↓</div>'
        '  <div class="flowchart-step">Query Rewrite</div>'
        '  <div class="flowchart-arrow">↓</div>'
        '  <div class="flowchart-step">Hybrid Retrieval (FAISS + BM25)</div>'
        '  <div class="flowchart-arrow">↓</div>'
        '  <div class="flowchart-step">Cross-Encoder Reranking</div>'
        '  <div class="flowchart-arrow">↓</div>'
        '  <div class="flowchart-step">Selected Grounding Context</div>'
        '  <div class="flowchart-arrow">↓</div>'
        '  <div class="flowchart-step">Gemini ReAct Tool Loop</div>'
        '  <div class="flowchart-arrow">↓</div>'
        '  <div class="flowchart-step">Final Synthesized Response</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Accordion 1: ReAct Steps
    with st.expander("📝 ReAct Reasoning Steps", expanded=True):
        for step in trace_steps:
            step_idx = step.get("step_index", 1)
            thought = step.get("thought", "").strip()
            tool_name = step.get("tool_name")
            tool_args = step.get("tool_args")
            observation = step.get("observation", "")
            
            st.markdown(f"**Step {step_idx}:** " + (f"Tool Call: `{tool_name}`" if tool_name else "Final Response"))
            st.markdown(f"*Thought:* {thought}")
            if tool_name:
                st.markdown(f"*Arguments:* `{tool_args}`")
                st.text_area("Observation", value=observation, height=100, key=f"trace_obs_{step_idx}")
            st.markdown("---")
            
    # Accordion 2: Retrieval Scores
    if retrieved_chunks:
        with st.expander("📊 Retrieval Chunk Scores", expanded=False):
            chunk_data = []
            for idx, c in enumerate(retrieved_chunks, 1):
                v_id = c.get("video_id", "")
                meta = rag_manager.video_metadata_map.get(v_id, {})
                title = meta.get("title", "Unknown Video")
                time_str = format_time(c.get("start_time", 0.0))
                
                chunk_data.append({
                    "Rank": idx,
                    "Video": title[:20] + "...",
                    "Time": time_str,
                    "FAISS": f"{c.get('dense_score', 0.0):.4f}",
                    "BM25": f"{c.get('sparse_score', 0.0):.4f}",
                    "Hybrid": f"{c.get('hybrid_score', 0.0):.4f}",
                    "Rerank": f"{c.get('rerank_score', 0.0):.4f}"
                })
            st.dataframe(chunk_data, use_container_width=True)
            
    # Accordion 3: Performance Metrics
    if retrieved_chunks and len(retrieved_chunks) > 0 and "dense_latency_ms" in retrieved_chunks[0]:
        first_chunk = retrieved_chunks[0]
        with st.expander("⚡ Latency & Specs", expanded=False):
            st.metric("Hybrid Search Latency", f"{first_chunk.get('dense_latency_ms', 0.0):.2f} ms")
            st.metric("Reranking Latency", f"{first_chunk.get('rerank_latency_ms', 0.0):.2f} ms")
            st.metric("Total Latency", f"{first_chunk.get('total_latency_ms', 0.0):.2f} ms")
            
    st.markdown('</div>', unsafe_allow_html=True)
