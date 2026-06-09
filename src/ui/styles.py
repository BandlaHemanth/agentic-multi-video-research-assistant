"""
styles.py — SaaS Light Theme CSS configurations.
Provides off-white page background, white cards, purple gradients, hover scaling animations,
ChatGPT chat alignment templates, and custom tab underlines.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* ──────────────────────────────────────────────────────────────── */
    /* 1. GLOBAL COLORS, FONTS & ACCENTS */
    /* ──────────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b !important; /* Slate 800 */
        background-color: #f8fafc !important; /* Premium off-white */
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #0f172a !important; /* Slate 900 */
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Set maximum reading content width & padding */
    .block-container {
        max-width: 1300px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 2. SIDEBAR STYLE (WHITE GLASSMORPHIC WITH SOFT DROPSHADOW) */
    /* ──────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.03) !important;
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #0f172a !important;
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 3. PREMIUM WHITE CARDS (SHADOWS, ROUNDED CORNERS & FADE-IN) */
    /* ──────────────────────────────────────────────────────────────── */
    .glass-card {
        background: #ffffff !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04), 0 8px 10px -6px rgba(15, 23, 42, 0.02) !important;
        animation: fadeIn 0.4s ease-out;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.08), 0 10px 10px -6px rgba(15, 23, 42, 0.04) !important;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 4. RED DEMO FALLBACK BANNER (PREMIUM ALERT) */
    /* ──────────────────────────────────────────────────────────────── */
    .fallback-banner {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 500 !important;
        color: #b91c1c !important; /* Crimson dark */
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08) !important;
        animation: pulse 2.5s infinite;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.003); box-shadow: 0 6px 16px rgba(239, 68, 68, 0.12); }
        100% { transform: scale(1); }
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 5. FLOWCHART & TRACING LOGS */
    /* ──────────────────────────────────────────────────────────────── */
    .trace-flowchart-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0;
        padding: 1.5rem;
        background: #f8fafc;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
    }

    .flowchart-step {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        font-weight: 600;
        color: #7c5cfc; /* Purple */
        text-align: center;
        min-width: 200px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
        transition: transform 0.2s ease;
    }
    
    .flowchart-step:hover {
        transform: scale(1.02);
        border-color: #7c5cfc;
    }

    .flowchart-arrow {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: bold;
    }

    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 6px;
        font-family: 'Fira Code', monospace;
    }

    .badge-dense { background-color: rgba(124, 92, 252, 0.08); color: #7c5cfc; border: 1px solid rgba(124, 92, 252, 0.2); }
    .badge-sparse { background-color: rgba(14, 165, 233, 0.08); color: #0ea5e9; border: 1px solid rgba(14, 165, 233, 0.2); }
    .badge-hybrid { background-color: rgba(168, 85, 247, 0.08); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.2); }
    .badge-rerank { background-color: rgba(34, 197, 94, 0.08); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.2); }

    /* ──────────────────────────────────────────────────────────────── */
    /* 6. CHAT ASSISTANT COMPONENT Redesign (CHATGPT STYLE) */
    /* ──────────────────────────────────────────────────────────────── */
    .chat-bubble-container {
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
        margin-bottom: 2rem;
        width: 100%;
        animation: fadeIn 0.3s ease-out;
    }

    .chat-message-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.5rem;
    }

    .chat-message-row.user-row {
        justify-content: flex-end;
    }

    .chat-message-row.assistant-row {
        justify-content: flex-start;
    }

    .chat-bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }

    .user-bubble {
        background: #7c5cfc !important; /* Accent purple */
        color: #ffffff !important;
        border-radius: 18px 18px 2px 18px !important;
        text-align: left;
    }

    .user-bubble p, .user-bubble li, .user-bubble ul, .user-bubble ol, .user-bubble code {
        color: #ffffff !important;
    }

    .assistant-bubble {
        background: #f1f5f9 !important; /* Soft gray-blue */
        color: #1e293b !important;
        border-radius: 18px 18px 18px 2px !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Small action buttons under bubbles */
    .bubble-actions-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.25rem;
        margin-left: 0.5rem;
        justify-content: flex-start;
    }
    
    .user-row + .bubble-actions-row {
        justify-content: flex-end;
        margin-right: 0.5rem;
        margin-left: 0;
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 7. STREAMLIT INTERACTIVE ELEMENTS OVERRIDES */
    /* ──────────────────────────────────────────────────────────────── */
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #7c5cfc !important;
        box-shadow: 0 0 0 3px rgba(124, 92, 252, 0.15) !important;
    }

    /* Buttons (Purple Gradients, Scaling) */
    .stButton>button {
        background: linear-gradient(135deg, #7c5cfc 0%, #5b3fd2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 12px rgba(124, 92, 252, 0.15) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton>button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(124, 92, 252, 0.25) !important;
        filter: brightness(1.05) !important;
    }
    
    .stButton>button:active {
        transform: scale(0.98) !important;
    }

    /* Red Remove / Wastebasket button inside sidebar */
    .remove-btn button {
        background: #fee2e2 !important;
        color: #ef4444 !important;
        border: 1px solid #fca5a5 !important;
        box-shadow: none !important;
        font-size: 0.85rem !important;
        padding: 0.35rem 0.6rem !important;
    }

    .remove-btn button:hover {
        background: #ef4444 !important;
        color: #ffffff !important;
        border-color: #ef4444 !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15) !important;
        transform: scale(1.05) !important;
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 8. ACTIVE TAB PURPLE UNDERLINE OVERRIDES */
    /* ──────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent !important;
        border-bottom: 2px solid #e2e8f0 !important;
        padding-bottom: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        color: #64748b !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.25rem !important;
        transition: color 0.15s ease !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #7c5cfc !important;
        font-weight: 600 !important;
        border-bottom: 3px solid #7c5cfc !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #7c5cfc !important;
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* 9. METRICS & THUMBNAIL LISTINGS */
    /* ──────────────────────────────────────────────────────────────── */
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .video-sidebar-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s ease;
    }
    
    .video-sidebar-card:hover {
        border-color: #7c5cfc;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
"""

def load_custom_css():
    """Injects custom CSS styling classes into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
