# 🎬 Agentic Multi-Video Research Assistant

An advanced, production-grade agentic research application designed to ingest, transcribe, index, search, and analyze YouTube videos. Backed by a hybrid dense-sparse RAG pipeline, local Cross-Encoder reranking, and a conversational ReAct agent loop powered by Google's Gemini 2.5 Flash.

---

## 📖 Overview

The **Agentic Multi-Video Research Assistant** provides researchers, students, and content creators with an automated platform to query and extract insights from YouTube video transcripts. By combining vector semantic search (FAISS) with lexical keyword matching (BM25), the application ensures robust query retrieval and precise grounding. In cases where Gemini API is unavailable or rate-limited, a local fallback system is used to generate deterministic transcript-grounded answers.

---

## 🏗️ Architecture & Technical Pipeline

The flowchart below represents the data ingestion, RAG retrieval, and agent reasoning pipelines:

```mermaid
graph TD
    A[YouTube URL] --> B(Ingestion Engine)
    B --> C{Transcripts Available?}
    C -- Yes (EN or Original Lang) --> D[Ingest Transcripts]
    C -- No --> E[Download Audio via yt-dlp]
    E --> F[ASR via faster-whisper CPU int8 / GPU fp16]
    F --> D
    D --> G[Timestamped Chunking]
    G --> H[Vector Store FAISS]
    G --> I[Lexical Index BM25]
    J[User Query] --> K[Hybrid Retrieval Search]
    H --> K
    I --> K
    K --> L[Min-Max Score Fusion]
    L --> M[Reranking via Cross-Encoder]
    M --> N[ReAct Agent Loop - Gemini 2.5 Flash]
    N --> O[Cited Response with Clickable Timestamps]
```

1. **Ingestion Engine:** Fetches metadata using `yt-dlp` and checks for manual or auto-generated transcripts using `youtube-transcript-api`.
2. **ASR Fallback:** If captions are disabled, it downloads the audio track and transcribes it locally using `faster-whisper` (GPU float16 if available, otherwise CPU int8).
3. **Semantic Chunking:** Groups segments into overlapping chunks keeping track of start/end timestamps.
4. **Hybrid Indexing:** Feeds chunks to `models/gemini-embedding-001` in a single batched call to construct a dense vector FAISS index, and tokenizes text to build a sparse lexical BM25 index.
5. **Hybrid Retrieval:** Blends FAISS and BM25 scores using min-max scaling with a custom `alpha` factor.
6. **Cross-Encoder Reranking:** Re-evaluates retrieval candidates using the local `ms-marco-MiniLM-L-6-v2` cross-encoder to select the top 5 most relevant grounding passages.
7. **ReAct Agent Loop:** A conversational agent loop (`VideoResearchAgent`) executing in-context tool calls (`search_videos`, `get_video_details`, `summarize_video`) to retrieve facts, verify details, or summarize content.
8. **Cited Answers:** Emits responses using clickable format: `[Video Title - MM:SS]`, which links directly to the corresponding YouTube timestamp.

---

## 🌟 Core Features

* **YouTube Video Ingestion & Transcript Extraction:** Automatically extracts manual or auto-generated transcripts. If unavailable, falls back to local Whisper ASR transcription.
* **Hybrid RAG Pipeline:** Combines dense semantic search (FAISS) and sparse lexical search (BM25) with min-max score blending.
* **Cross-Encoder Reranking:** Leverages the local `ms-marco-MiniLM-L-6-v2` model to rerank the top candidates and improve precision.
* **AI Chat Assistant:** Conversational agent loop (`VideoResearchAgent`) executing ReAct reasoning with citations and clickable timestamps.
* **Transcript-Grounded Video Summarization:** Provides comprehensive summaries structured with main topics, key points, takeaways, and notable quotes.
* **Transcript-Grounded Quiz Generation:** Generates interactive 5-question multiple-choice quizzes strictly grounded in the video's transcript.
* **Gemini API Key Override:** Allows users to paste their own Gemini API key in the UI settings sidebar to override the default credentials.
* **Local Fallback when Gemini is Unavailable:** Instantly switches to a deterministic local fallback system (using hybrid search and cross-encoding) if Gemini API returns HTTP 429 rate limits or fails.
* **Intelligent Query Rewriting:** Automatically preprocesses and rewrites vague queries (e.g., "summarize the video") to target the single indexed video.
* **Batched Embedding Generation:** Batches all transcript chunks into a single embedding API call for improved efficiency and quota conservation.
* **Execution Trace (Debug Mode):** Toggle debug mode to reveal intermediate steps (thoughts, tool arguments, latency, dense/sparse/rerank scores).

---

## 🛠️ Tech Stack

* **Front-end UI:** Streamlit (with custom premium CSS styling)
* **AI Orchestration & Agents:** LangChain Core, Google GenAI SDK (Gemini 2.5 Flash)
* **Embeddings:** Gemini API (`models/gemini-embedding-001`)
* **Vector Indexing:** FAISS CPU
* **Lexical Indexing:** Rank-BM25
* **Reranking:** Sentence-Transformers (Cross-Encoder `ms-marco-MiniLM-L-6-v2`)
* **ASR Transcriptions:** faster-whisper, PyTorch, imageio-ffmpeg
* **Scraping & Subtitles:** yt-dlp, youtube-transcript-api
* **Evaluation:** Ragas, HuggingFace Datasets

---

## ⚠️ Known Limitations

* **Playlist URLs are not currently supported:** The current version only supports single YouTube video URLs.
* **Ingestion Failures in Cloud Deployments:** Ingesting some videos may fail on cloud environments (like Streamlit Community Cloud) due to transcript availability blocks or server audio download restrictions.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- **FFmpeg:** Required for audio extraction during ASR fallback.
  - *Windows:* Install via Chocolatey (`choco install ffmpeg`) or download static binaries and add to system PATH.
  - *macOS:* `brew install ffmpeg`
  - *Linux:* `sudo apt-get install ffmpeg`

### Step 1: Clone and Enter the Project Directory
```bash
git clone <repository_url>
cd agentic-video-research
```

### Step 2: Set Up a Virtual Environment
```bash
python -m venv .venv
# Activate on Windows (cmd)
.venv\Scripts\activate
# Activate on Windows (PowerShell)
& .venv/Scripts/Activate.ps1
# Activate on macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file at the root of the project:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
WHISPER_ASR_MODEL=base
```
*Note: If no Google API key is supplied, the application will run in local fallback mode with static and RAG-based responses.*

---

## 💻 Usage

### Launching the Streamlit Web Application
Ensure your virtual environment is active:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Running CLI Verification Scripts
Verify the ingestion and fallback pipelines:
```bash
# Verify transcript fetch priority and ASR fallback
python ../test_fallback.py

# Verify indexing and retrieval
python test_retrieval.py --url "https://www.youtube.com/watch?v=ySEx_BqVx8A" --query "what is an API"

# Verify answer generation and Ragas evaluation
python test_generation.py

# Verify agentic ReAct loop and tool calls
python test_agent.py
```
