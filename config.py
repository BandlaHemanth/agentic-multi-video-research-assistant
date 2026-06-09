"""
config.py — Centralized configuration for the Agentic Multi-Video Research Assistant.
All tunable parameters live here. Import this module everywhere instead of hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
LOG_DIR = DATA_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────
# GEMINI MODELS
# ─────────────────────────────────────────────
GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_DIM: int = 3072  # Dimension for models/gemini-embedding-001

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
CHUNK_SIZE: int = 1000          # Characters per chunk
CHUNK_OVERLAP: int = 200        # Overlap between consecutive chunks
MIN_CHUNK_SIZE: int = 100       # Discard chunks shorter than this

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
FAISS_TOP_K: int = 20           # Number of candidates from FAISS
BM25_TOP_K: int = 20            # Number of candidates from BM25
HYBRID_ALPHA: float = 0.7       # Weight for dense retrieval (1-alpha for sparse)
FINAL_CANDIDATES: int = 20      # Total merged candidates passed to reranker

# ─────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K: int = 5           # Final chunks sent to LLM after reranking

# ─────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.2    # Low temperature for factual answers
LLM_MAX_TOKENS: int = 2048

# ─────────────────────────────────────────────
# QUIZ
# ─────────────────────────────────────────────
QUIZ_QUESTIONS_PER_LEVEL: int = 5   # Questions per difficulty level

# ─────────────────────────────────────────────
# EVALUATION (RAGAS)
# ─────────────────────────────────────────────
RAGAS_EVAL_LLM_MODEL: str = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# ASR (AUTOMATIC SPEECH RECOGNITION)
# ─────────────────────────────────────────────
WHISPER_ASR_MODEL: str = os.getenv("WHISPER_ASR_MODEL", "base")

# ─────────────────────────────────────────────
# UI / APP
# ─────────────────────────────────────────────
APP_TITLE: str = "🎬 Agentic Multi-Video Research Assistant"
APP_ICON: str = "🎬"
APP_VERSION: str = "1.0.0"
