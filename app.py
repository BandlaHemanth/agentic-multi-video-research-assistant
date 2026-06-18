"""
app.py — Lightweight entry point and UI orchestrator (SaaS Redesign).
Sets page layout, injects custom light-mode stylesheets, maintains shared session states,
mounts sidebar configurations, and delegates tab content rendering.
"""

import os
import streamlit as st
import google.genai as google_genai

# 1. Page Configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="🎬 Agentic Multi-Video Research Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.core.rag import HybridRAGManager
from src.core.agent import VideoResearchAgent
from src.ui.styles import load_custom_css
from src.ui.sidebar import render_sidebar
from src.ui.tab_chat import render_chat_tab
from src.ui.tab_compare import render_compare_tab
from src.ui.tab_summaries import render_summaries_tab
from src.ui.tab_quiz import render_quiz_tab
from src.ui.tab_evaluation import render_evaluation_tab

# 2. Ingest styling
load_custom_css()

# 3. Cache RAG Manager and Agent as shared resources
@st.cache_resource
def get_rag_manager() -> HybridRAGManager:
    return HybridRAGManager()

@st.cache_resource
def get_agent() -> VideoResearchAgent:
    rag = get_rag_manager()
    return VideoResearchAgent(rag_manager=rag)

# Initialize resources
rag_manager = get_rag_manager()
agent = get_agent()

# 4. Render Sidebar (determines default or override active API key)
sidebar_opts = render_sidebar(rag_manager)
active_key = sidebar_opts["api_key"]

# 5. Sync API key changes dynamically to environment variables and clients
import logging
logger = logging.getLogger(__name__)
if active_key:
    logger.info(f"[DEBUG] Syncing active key ({st.session_state.get('active_key_source', 'unknown source')}) globally.")
    print(f"[DEBUG] Syncing active key ({st.session_state.get('active_key_source', 'unknown source')}) globally.")
    os.environ["GOOGLE_API_KEY"] = active_key
    import google.generativeai as genai
    genai.configure(api_key=active_key)
    try:
        agent.client = google_genai.Client(api_key=active_key)
        logger.info("[DEBUG] google.genai Client configured successfully.")
        print("[DEBUG] google.genai Client configured successfully.")
    except Exception as e:
        logger.error(f"[DEBUG] Failed to configure Gemini Client: {e}")
        print(f"[DEBUG] Failed to configure Gemini Client: {e}")
        st.sidebar.error(f"Failed to configure Gemini Client: {e}")
        agent.client = None
else:
    logger.warning("[DEBUG] No active API key found. Clearing environment and client.")
    print("[DEBUG] No active API key found. Clearing environment and client.")
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    agent.client = None

# 6. Main App Header (SaaS Premium Redesign)
st.markdown('''
<div style="margin-bottom: 2rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 1.5rem; text-align: left;">
    <h1 style="font-size: 2.6rem; font-weight: 700; margin-bottom: 0.35rem; color: #0f172a;">🎬 Agentic Multi-Video Research Assistant</h1>
    <p style="color: #64748b; font-size: 1.1rem; font-weight: 400; margin: 0;">Search, summarize, compare and analyze video content with AI.</p>
</div>
''', unsafe_allow_html=True)

# Scenario D: If no key is set anywhere, display a clear, user-friendly warning message
if not active_key:
    st.warning("⚠️ **Gemini API Key Required**: Please configure a valid Gemini API Key in the sidebar to enable the search, chat, comparison, summary, and quiz features.")

# 7. Render Layout Tabs
tab_labels = [
    "💬 Chat Assistant", 
    "🔍 Compare Videos", 
    "📝 Summaries", 
    "🎯 Quiz Room", 
    "📊 Evaluation Dashboard"
]
tab_chat, tab_compare, tab_summaries, tab_quiz, tab_evaluation = st.tabs(tab_labels)

with tab_chat:
    render_chat_tab(agent, rag_manager, sidebar_opts["api_key"])

with tab_compare:
    render_compare_tab(agent, rag_manager)

with tab_summaries:
    render_summaries_tab(agent, rag_manager)

with tab_quiz:
    render_quiz_tab(agent, rag_manager)

with tab_evaluation:
    render_evaluation_tab(agent, rag_manager)
