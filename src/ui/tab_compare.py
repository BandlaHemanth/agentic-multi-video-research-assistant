"""
tab_compare.py — Video Comparison UI tab (SaaS Redesign).
Allows selecting multiple videos, writing a comparative query, executing agent
searches restricted to specific video contexts, and presenting side-by-side or tabular summaries.
"""

import streamlit as st
import logging
from typing import List

from src.core.agent import VideoResearchAgent
from src.core.rag import HybridRAGManager
from src.ui.utils import parse_citations, render_trace_panel

logger = logging.getLogger(__name__)

def render_compare_tab(agent: VideoResearchAgent, rag_manager: HybridRAGManager):
    """
    Renders the Video Comparison interface.
    """
    st.subheader("🔍 Compare Videos")
    st.markdown("*Select multiple indexed videos and input a research query to generate comparative analysis tables.*")
    
    # Check if we have videos
    indexed_videos = rag_manager.video_metadata_map
    if not indexed_videos:
        st.info("No videos currently indexed. Please index videos in the sidebar first!")
        return
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    # Multi-select dropdown
    selected_ids = st.multiselect(
        "Select Videos to Compare",
        options=list(indexed_videos.keys()),
        format_func=lambda v_id: indexed_videos[v_id].get("title", f"Video {v_id}"),
        help="Choose two or more videos to compare their transcript contexts."
    )
    
    # Comparative Query Input
    comparison_query = st.text_area(
        "Comparative Research Question",
        placeholder="Compare the architecture, dense/sparse search parameters, or specific models mentioned in these videos...",
        height=100
    )
    
    # Run Button
    compare_button = st.button("Generate Comparison Analysis", key="compare_execute_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if compare_button:
        if not selected_ids:
            st.error("Please select at least one video to compare.")
            return
        if not comparison_query.strip():
            st.error("Please enter a comparative question.")
            return
            
        selected_titles_str = ", ".join([
            f"'{indexed_videos[v_id].get('title')}' (ID: {v_id})"
            for v_id in selected_ids
        ])
        
        prompt = (
            f"You are comparing the following specific videos: {selected_titles_str}.\n"
            f"Analyze their transcripts to answer the following research question: '{comparison_query}'.\n"
            f"Ground your answer STRICTLY in the transcripts of ONLY the selected videos: {selected_ids}.\n"
            f"Structure your response logically. If applicable, represent your analysis in a clean markdown table "
            f"detailing key comparative aspects. Ensure every claim is cited using [Video Title - MM:SS]."
        )
        
        with st.spinner("Analyzing transcripts and compiling comparative table..."):
            try:
                result = agent.run(prompt)
                
                # Render inside card
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Comparison Analysis Results")
                
                # Replace citations
                formatted_ans = parse_citations(result.answer, indexed_videos)
                st.markdown(formatted_ans)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # If debug mode enabled, render trace
                if st.session_state.get("debug_mode", False) and result.trace_steps:
                    with st.expander("🔍 Inspection & Trace Logs", expanded=False):
                        render_trace_panel([ts.__dict__ for ts in result.trace_steps], [c.__dict__ for c in result.retrieved_chunks], rag_manager)
                        
            except Exception as e:
                st.error(f"Failed to generate comparison: {e}")
                logger.error(f"Comparison generation failed: {e}", exc_info=True)
