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
    
    st.warning("🚧 **Compare Videos**\n\nThis feature is currently under development and will be available in a future update.")
    st.markdown("*Multi-video comparison is being actively developed and will be enabled in an upcoming release.*")
