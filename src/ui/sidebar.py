"""
sidebar.py — Sidebar configuration and callback handler (SaaS redesign).
Renders white card components for Gemini API key, URL ingestion, indexed video lists with
YouTube thumbnails, delete triggers, index rebuilding, and database stats.
"""

import os
import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any

from config import GOOGLE_API_KEY
from src.core.ingestion import ingest_video, extract_playlist_video_urls, TRANSCRIPT_CACHE_DIR, load_cached_video_data
from src.core.rag import HybridRAGManager, format_time

logger = logging.getLogger(__name__)

def render_sidebar(rag_manager: HybridRAGManager) -> Dict[str, Any]:
    """
    Renders the SaaS-redesigned sidebar components.
    
    Args:
        rag_manager (HybridRAGManager): The active RAG manager instance.
        
    Returns:
        Dict[str, Any]: Dict containing keys:
            - 'api_key': Active Gemini API key string
            - 'debug_mode': Boolean debug configuration flag
    """
    # Sidebar Header
    st.sidebar.markdown("## ⚙️ SaaS Settings")
    st.sidebar.markdown("Manage your API keys, ingest content, and monitor databases.")
    st.sidebar.markdown("---")
    
    # ────────────────────────────────────────────────────────────────
    # CARD 1: GEMINI API KEY CONFIGURATION
    # ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔑 API Authentication")
    
    # Helper validation function
    def validate_gemini_api_key(key: str) -> bool:
        if not key or not isinstance(key, str):
            st.session_state["api_key_error"] = "API Key cannot be empty."
            return False
        key = key.strip().strip("'").strip('"')
        obfuscated_key = "None"
        if key:
            if len(key) > 8:
                obfuscated_key = key[:4] + "..." + key[-4:]
            else:
                obfuscated_key = "..."
        print(f"[VALIDATOR] Obfuscated key: {obfuscated_key}")
        print(f"[VALIDATOR] Length: {len(key)}")
        if len(key) < 20:
            st.session_state["api_key_error"] = "Invalid key length. API key must be at least 20 characters long."
            return False
        try:
            import google.genai as google_genai
            import google.genai.errors as errors
            import httpx
            import traceback

            print("[VALIDATOR] Creating google.genai.Client...")
            client = google_genai.Client(api_key=key)
            print("[VALIDATOR] Client created successfully")
            
            print("[VALIDATOR] Calling list(client.models.list())...")
            list(client.models.list())
            print("[VALIDATOR] list(client.models.list()) completed successfully!")
            
            st.session_state["api_key_error"] = None
            return True
        except Exception as e:
            import traceback
            print("[VALIDATOR] Exception occurred:")
            print(type(e).__name__)
            print(str(e))
            traceback.print_exc()
            st.session_state["api_key_error"] = f"{type(e).__name__}: {str(e)}"
            return False

    # Check for default key in Streamlit secrets or env vars
    def get_default_key() -> str:
        try:
            if st.secrets and "GOOGLE_API_KEY" in st.secrets:
                val = st.secrets["GOOGLE_API_KEY"]
                if isinstance(val, str) and val.strip():
                    return val.strip().strip("'").strip('"')
        except Exception:
            pass
        return os.getenv("GOOGLE_API_KEY", "").strip().strip("'").strip('"')

    default_key = get_default_key()
    has_default = bool(default_key)
    
    override_key_input = ""
    
    if has_default:
        st.sidebar.success("✅ Default Gemini API Key Loaded")
        with st.sidebar.expander("Use a different API key"):
            override_key_input = st.text_input(
                "Override Gemini API Key",
                type="password",
                placeholder="Enter Gemini API key...",
                help="Provide your own Gemini API Key to override the default key."
            ).strip().strip("'").strip('"')
    else:
        override_key_input = st.sidebar.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter Gemini API key...",
            help="Input your Gemini API Key to enable responses."
        ).strip().strip("'").strip('"')
        
    # Handle validation and session state updating
    if override_key_input:
        last_validated = st.session_state.get("last_validated_key", "")
        if override_key_input != last_validated:
            with st.sidebar:
                with st.spinner("Validating API key..."):
                    is_valid = validate_gemini_api_key(override_key_input)
                    st.session_state["last_validated_key"] = override_key_input
                    st.session_state["is_key_valid"] = is_valid
                    if is_valid:
                        st.session_state["api_key_override"] = override_key_input
                        st.rerun()
                    else:
                        st.session_state["api_key_override"] = ""
        
        if st.session_state.get("is_key_valid") is False:
            err_msg = st.session_state.get("api_key_error", "Invalid API Key. Please check your credentials.")
            st.sidebar.error(f"❌ {err_msg}")
    else:
        if st.session_state.get("api_key_override", ""):
            st.session_state["api_key_override"] = ""
            st.session_state["last_validated_key"] = ""
            st.session_state["is_key_valid"] = None
            st.rerun()

    # Determine current status
    active_key = st.session_state.get("api_key_override", "").strip() or default_key
    key_source = "user key" if st.session_state.get("api_key_override", "").strip() else ("default key" if default_key else "none")

    # Render Status Card
    if active_key:
        status_text = "Connected using user key" if key_source == "user key" else "Connected using default key"
        status_color = "#15803d"  # green-700
        status_bg = "#f0fdf4"     # green-50
        status_border = "#bbf7d0" # green-200
    else:
        status_text = "No API key configured"
        status_color = "#b91c1c"  # red-700
        status_bg = "#fef2f2"     # red-50
        status_border = "#fecaca" # red-200

    st.sidebar.markdown(
        f'<div class="glass-card" style="background-color: {status_bg}; border: 1px solid {status_border}; padding: 12px; margin-bottom: 15px; border-radius: 12px !important;">'
        f'  <span style="font-weight: 600; color: {status_color}; font-size: 0.85rem;">Gemini Status:</span><br>'
        f'  <span style="color: {status_color}; font-size: 0.82rem;">{status_text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")

    # ────────────────────────────────────────────────────────────────
    # CARD 2: VIDEO & PLAYLIST INGESTION
    # ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📥 Content Ingest")
    
    # Single Video Form
    with st.sidebar.form("video_ingest_form", clear_on_submit=True):
        video_url = st.text_input(
            "Paste a YouTube video URL (playlists not supported)", 
            placeholder="Enter a single YouTube video URL"
        )
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-top: -10px; margin-bottom: 10px;'>"
            "Note: Playlist URLs are not supported in the current version."
            "</p>",
            unsafe_allow_html=True
        )
        submit_video = st.form_submit_button("Index Video")
        
    if submit_video and video_url.strip():
        url_stripped = video_url.strip()
        if "list=" in url_stripped:
            st.sidebar.warning("⚠️ Playlist URLs are not supported yet. Please provide a single YouTube video URL.")
        else:
            with st.sidebar:
                status_container = st.empty()
                progress_messages = []
                
                def progress_cb(msg):
                    progress_messages.append(msg)
                    # Formulate visual list with checklists and spinner
                    status_container.markdown(
                        '<div class="glass-card" style="border-left: 4px solid #7c5cfc; padding: 12px; margin-bottom: 12px; font-size: 0.85rem;">'
                        '  <strong>Ingestion Pipeline:</strong><br>' +
                        "<br>".join([f"✓ {m}" for m in progress_messages[:-1]] + [f"🔄 {progress_messages[-1]}"]) +
                        '</div>',
                        unsafe_allow_html=True
                    )
                
                try:
                    # 1. Metadata and Transcript retrieval (YouTube api or Whisper fallback)
                    video_data = ingest_video(url_stripped, progress_callback=progress_cb)
                    
                    # 2. Split chunking
                    progress_cb("Chunking transcript segments...")
                    
                    # 3. Dense FAISS + Sparse BM25 indexing
                    progress_cb("Generating embeddings & building index...")
                    rag_manager.add_video(video_data)
                    
                    # 4. Success state
                    progress_cb("Indexing completed.")
                    
                    st.success(f"Indexed: {video_data.metadata.title[:30]}...")
                    st.toast(f"Indexed: {video_data.metadata.title[:30]}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest video: {e}")
                    logger.error(f"Video ingestion failed: {e}", exc_info=True)

    st.sidebar.markdown("---")

    # ────────────────────────────────────────────────────────────────
    # CARD 3: INDEXED VIDEOS (WITH THUMBNAILS & DELETIONS)
    # ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🗄️ Indexed Database")
    
    indexed_videos = rag_manager.video_metadata_map
    if not indexed_videos:
        st.sidebar.info("No videos indexed yet.")
    else:
        for video_id, meta in list(indexed_videos.items()):
            title = meta.get("title", f"Video {video_id}")
            duration_str = format_time(meta.get("duration", 0))
            
            # YouTube Thumbnail image URL
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            
            # Render visual listing card
            st.sidebar.markdown(f'<div class="video-sidebar-card">', unsafe_allow_html=True)
            
            # Thumbnail image
            st.sidebar.image(thumbnail_url, use_container_width=True)
            
            # Title & Metadata
            st.sidebar.markdown(f"**{title}**")
            lang = meta.get("language", "English")
            src_type = meta.get("transcript_source", "manual")
            src_label = "manual"
            if src_type == "auto_generated":
                src_label = "auto-gen"
            elif src_type == "whisper_fallback":
                src_label = "whisper ASR"
            st.sidebar.markdown(f"*Duration: {duration_str} | Lang: {lang} | Src: {src_label}*")
            
            # Inline column delete button
            st.sidebar.markdown('<div class="remove-btn">', unsafe_allow_html=True)
            if st.sidebar.button("🗑️ Remove Video", key=f"del_{video_id}", help="Delete video and rebuild index"):
                with st.sidebar:
                    with st.spinner("Deleting chunks and rebuilding index..."):
                        try:
                            rag_manager.remove_video(video_id)
                            st.success("Removed video!")
                            st.rerun()
                        except Exception as err:
                            st.error(f"Error: {err}")
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            st.sidebar.markdown(" ")

    st.sidebar.markdown("---")

    # ────────────────────────────────────────────────────────────────
    # CARD 4: REBUILD & SETTINGS
    # ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🛠️ Maintenance")
    
    if st.sidebar.button("Rebuild Index", key="sidebar_rebuild_btn", help="Recreate index from cached transcript json files"):
        with st.sidebar:
            with st.spinner("Rebuilding indexes from cached files..."):
                try:
                    cache_files = list(Path(TRANSCRIPT_CACHE_DIR).glob("*.json"))
                    if not cache_files:
                        st.info("No cache found to rebuild.")
                    else:
                        rag_manager.video_metadata_map.clear()
                        rag_manager.chunks.clear()
                        rebuild_count = 0
                        for filepath in cache_files:
                            v_id = filepath.stem
                            cached_data = load_cached_video_data(v_id)
                            if cached_data:
                                rag_manager.add_video(cached_data)
                                rebuild_count += 1
                        st.success(f"Rebuilt index for {rebuild_count} videos!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to rebuild: {e}")
                    logger.error(f"Index rebuild failed: {e}", exc_info=True)

    # Toggle Debug Mode
    debug_mode = st.sidebar.toggle("Debug Mode", value=st.session_state.get("debug_mode", False), key="sidebar_debug_toggle")
    st.session_state["debug_mode"] = debug_mode
    
    # Stats Card
    total_videos = len(rag_manager.video_metadata_map)
    total_chunks = len(rag_manager.chunks)
    
    st.sidebar.markdown(" ")
    col_stat1, col_stat2 = st.sidebar.columns(2)
    with col_stat1:
        st.markdown(f'<div class="metric-label">Videos</div><div class="metric-value">{total_videos}</div>', unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f'<div class="metric-label">Chunks</div><div class="metric-value">{total_chunks}</div>', unsafe_allow_html=True)
        
    # System Information Card
    import torch
    from config import WHISPER_ASR_MODEL
    gpu_avail = torch.cuda.is_available()
    device_name = "GPU (CUDA)" if gpu_avail else "CPU"
    quant_format = "float16" if gpu_avail else "int8 (quantized)"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ System Information")
    st.sidebar.markdown(
        f'<div class="glass-card" style="padding: 12px; margin-bottom: 0px; font-size: 0.82rem; line-height: 1.4; border-radius: 12px !important;">'
        f'  <strong>ASR Model:</strong> <code>{WHISPER_ASR_MODEL}</code><br>'
        f'  <strong>ASR Device:</strong> <code>{device_name}</code><br>'
        f'  <strong>ASR Format:</strong> <code>{quant_format}</code><br>'
        f'  <strong>Index Type:</strong> <code>Hybrid RAG</code>'
        f'</div>',
        unsafe_allow_html=True
    )
        
    return {
        "api_key": active_key,
        "debug_mode": debug_mode
    }
