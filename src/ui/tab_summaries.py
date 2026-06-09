"""
tab_summaries.py — Video Summaries and Metadata viewer UI tab (SaaS Redesign).
Displays metadata cards for a selected video and generates / caches transcript-grounded summaries.
"""

import streamlit as st
import logging

from src.core.agent import VideoResearchAgent
from src.core.rag import HybridRAGManager, format_time

logger = logging.getLogger(__name__)

def render_summaries_tab(agent: VideoResearchAgent, rag_manager: HybridRAGManager):
    """
    Renders the Video Summaries interface.
    """
    st.subheader("📝 Video Summaries")
    st.markdown("*Select an indexed video to inspect its metadata details and generate a bulleted transcript summary.*")
    
    # Check if we have videos
    indexed_videos = rag_manager.video_metadata_map
    if not indexed_videos:
        st.info("No videos currently indexed. Please index videos in the sidebar first!")
        return
        
    # Selectbox for video
    selected_id = st.selectbox(
        "Select Video",
        options=list(indexed_videos.keys()),
        format_func=lambda v_id: indexed_videos[v_id].get("title", f"Video {v_id}"),
        help="Choose a video to see metadata and summaries."
    )
    
    if not selected_id:
        return
        
    meta = indexed_videos[selected_id]
    
    # ────────────────────────────────────────────────────────────────
    # METADATA CARDS DISPLAY (WHITE GLASS CARD)
    # ────────────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### 🎥 {meta.get('title')}")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f'<div class="metric-label">Uploader</div><div class="metric-value" style="font-size:1.15rem; color:#7c5cfc;">{meta.get("author", "Unknown")}</div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-label">Upload Date</div><div class="metric-value" style="font-size:1.15rem; color:#0ea5e9;">{meta.get("upload_date", "Unknown")}</div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-label">Duration</div><div class="metric-value" style="font-size:1.15rem;">{format_time(meta.get("duration", 0))}</div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-label">Views</div><div class="metric-value" style="font-size:1.15rem;">{meta.get("view_count", 0):,}</div>', unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────
    # SUMMARY DISPLAY
    # ────────────────────────────────────────────────────────────────
    if "summaries_cache" not in st.session_state:
        st.session_state.summaries_cache = {}
        
    cached_summary = st.session_state.summaries_cache.get(selected_id)
    
    st.markdown("### 📝 Transcript Summary")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if cached_summary:
        st.markdown(cached_summary)
    else:
        st.info("No summary generated yet for this video.")
        generate_button = st.button("Generate Summary", key=f"sum_btn_{selected_id}")
        
        if generate_button:
            with st.spinner("Analyzing transcript segments and synthesizing summary..."):
                try:
                    summary_raw = agent._summarize_video(selected_id)
                    summary_text = summary_raw.replace("Observation:\n", "").replace("Observation:", "").strip()
                    
                    st.session_state.summaries_cache[selected_id] = summary_text
                    st.markdown(summary_text)
                    st.toast("Summary generated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate video summary: {e}")
                    logger.error(f"Summary generation failed for video {selected_id}: {e}", exc_info=True)
    st.markdown('</div>', unsafe_allow_html=True)
