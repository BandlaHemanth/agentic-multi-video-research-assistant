"""
tab_quiz.py — Video Quizzes UI tab (SaaS Redesign).
Generates 5 multiple choice questions using Gemini (or a synthetic fallback) grounded in
video transcripts. Handles interactive choice selection, grading state, and display of answers.
"""

import json
import logging
import streamlit as st
from typing import Dict, List, Any

from src.core.agent import VideoResearchAgent
from src.core.rag import HybridRAGManager
from config import GEMINI_LLM_MODEL

logger = logging.getLogger(__name__)

# Predefined high-quality synthetic quizzes for the default architecture guide video (ySEx_BqVx8A)
DEFAULT_QUIZ_QUESTIONS = {
    "Easy": [
        {
            "question": "Which large language model is leveraged by the system for reasoning and RAG generation?",
            "options": ["Gemini 1.5 Pro", "Gemini 2.5 Flash", "GPT-4o Mini", "Claude 3.5 Sonnet"],
            "correct_answer_index": 1,
            "explanation": "The transcript explicitly mentions: 'For the primary language model we leverage the newly released Gemini 2.5 Flash model which excels at reasoning.'"
        },
        {
            "question": "What local library is used to index dense vector embeddings?",
            "options": ["ChromaDB", "Pinecone CPU", "FAISS CPU Flat IP", "Milvus Server"],
            "correct_answer_index": 2,
            "explanation": "According to the transcript, 'These dense vectors are indexed locally using a FAISS CPU Index Flat IP for rapid semantic retrieval.'"
        },
        {
            "question": "What is the primary uploader of the guide video according to synthetic details?",
            "options": ["Google AI", "Gemini Dev Community", "DeepMind Devs", "OpenSource Community"],
            "correct_answer_index": 1,
            "explanation": "The video uploader / author metadata is defined as 'Gemini Dev Community'."
        },
        {
            "question": "What format should citations follow in the generated answers?",
            "options": ["(Author, Year)", "[Video Title - MM:SS]", "[Video ID - Page Number]", "Footnote (1)"],
            "correct_answer_index": 1,
            "explanation": "The agent prompt and system guidelines require citations to follow the format: [Video Title - MM:SS]."
        },
        {
            "question": "Which local cross-encoder model is used to filter out noise and rerank candidates?",
            "options": ["ms-marco-MiniLM-L-6-v2", "bert-base-uncased", "all-MiniLM-L6-v2", "bge-reranker-large"],
            "correct_answer_index": 0,
            "explanation": "The guide transcript specifies: 'To filter out noise we feed the fused candidates into a local cross-encoder model called ms-marco-MiniLM-L-6-v2.'"
        }
    ],
    "Medium": [
        {
            "question": "What embedding model is used to generate the dense vector representation?",
            "options": ["models/embedding-001", "text-embedding-3-small", "models/gemini-embedding-001", "bge-large-en-v1.5"],
            "correct_answer_index": 0,
            "explanation": "The guide transcript states: 'Our system uses models/embedding-001 to generate 768-dimensional dense vector embeddings for search. (Note: config.py maps to 3072 dimension version).'"
        },
        {
            "question": "How are dense and sparse retrieval ranks combined in the hybrid search pipeline?",
            "options": ["Reciprocal Rank Fusion (RRF)", "Normalized min-max score blending with an alpha parameter", "Simple average addition", "Cosine score multiplication"],
            "correct_answer_index": 1,
            "explanation": "The transcript states: 'We then fuse these dense and sparse retrieval ranks using a min-max scoring method with a customizable alpha parameter.'"
        },
        {
            "question": "What is the tokenized representation used for exact keyword matches?",
            "options": ["FAISS index", "BM25 index", "Reranker pairs", "Gemini prompt tokens"],
            "correct_answer_index": 1,
            "explanation": "The transcript states: 'To cover exact keyword lookups we combine FAISS dense search with a sparse BM25 index built on tokenized chunks.'"
        },
        {
            "question": "How many final context passages does the cross-encoder filter the candidates down to?",
            "options": ["Top 10", "Top 3", "Top 5", "Top 2"],
            "correct_answer_index": 2,
            "explanation": "The transcript states: 'This reranker refines the ordering and narrows the selection down to the top five most relevant context passages.'"
        },
        {
            "question": "What happens when the user clicks '🗑️' next to a video in the sidebar?",
            "options": ["Only the metadata is hidden", "The video and its chunks are removed and the dense/sparse index is rebuilt", "The cache file on disk is deleted but index remains unchanged", "The application database is reset entirely"],
            "correct_answer_index": 1,
            "explanation": "Removing a video filters out chunks belonging to it, rebuilds the FAISS index and BM25 index from scratch, and persists the files."
        }
    ],
    "Hard": [
        {
            "question": "What dimension vector is generated by the models/embedding-001 model in the system config?",
            "options": ["768", "1536", "3072", "1024"],
            "correct_answer_index": 2,
            "explanation": "In config.py, the dense embedding dimension is explicitly configured as 3072."
        },
        {
            "question": "Why is the cross-encoder loaded lazily inside the HybridRAGManager?",
            "options": ["To avoid memory leaks", "To save overhead and speed up initial imports when the reranker is not used", "Because Streamlit doesn't support eager imports", "To bypass API rate limits"],
            "correct_answer_index": 1,
            "explanation": "The HybridRAGManager loads the CrossEncoder class lazily inside the @property decorator to avoid heavy startup model load times when doing ingestion or basic metadata operations."
        },
        {
            "question": "Which parameter scales the fusion weight between dense and sparse ranks in hybrid search?",
            "options": ["temperature", "alpha", "top_k", "overlap"],
            "correct_answer_index": 1,
            "explanation": "The alpha parameter determines the blend weight: alpha * dense_score + (1.0 - alpha) * sparse_score."
        },
        {
            "question": "How does the ingestion process handle missing transcripts or network block issues?",
            "options": ["It crashes and outputs stack traces", "It retrieves translation tables from Google Translate", "It falls back to generating high-quality synthetic transcript descriptions", "It skips the video completely and indexes next entry"],
            "correct_answer_index": 2,
            "explanation": "If YouTube transcripts block requests (e.g. in cloud VMs), ingestion.py falls back to generating synthetic transcripts (generate_synthetic_video_data) to maintain executable environments."
        },
        {
            "question": "What method is used to extract averaged scores from RAGAS EvaluationResult to avoid AttributeError errors?",
            "options": ["result.get('metric_name')", "result._repr_dict.get('metric_name')", "getattr(result, 'scores')", "result.average_scores()"],
            "correct_answer_index": 1,
            "explanation": "Ragas EvaluationResult objects do not expose a dictionary get() directly. We fetch the scores dictionary dynamically using result._repr_dict.get(metric_name) followed by NaN float validation."
        }
    ]
}

def generate_quiz_from_llm(agent: VideoResearchAgent, title: str, transcript: str, level: str) -> List[Dict[str, Any]]:
    """
    Asks Gemini to generate 5 multiple choice questions based on the video transcript.
    """
    prompt = f"""You are a university professor. Generate an interactive multiple-choice quiz of exactly 5 questions based on the following video transcript.
The quiz difficulty should be: {level}.

Your response MUST be a single JSON array of objects. Do NOT wrap it in ```json code fences or markdown blocks, return ONLY the raw JSON string.
Each question object MUST have the following structure:
{{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer_index": 0,
    "explanation": "Detailed explanation grounding the answer in the transcript."
}}

Ensure correct_answer_index is a 0-indexed integer (0, 1, 2, or 3) representing the index of the correct string in the options array.

Video Title: {title}
Transcript:
{transcript[:8000]}
"""
    try:
        client = agent._get_client()
        if client is None:
            raise ValueError("Gemini API client not initialized.")
        response = agent.call_with_retry(
            client.models.generate_content,
            model=GEMINI_LLM_MODEL,
            contents=prompt
        )
        
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        questions = json.loads(text)
        if isinstance(questions, list) and len(questions) == 5:
            return questions
        else:
            raise ValueError("LLM did not return exactly 5 questions.")
            
    except Exception as e:
        logger.error(f"Failed to generate quiz via LLM: {e}. Falling back to default questions.")
        return DEFAULT_QUIZ_QUESTIONS.get(level, DEFAULT_QUIZ_QUESTIONS["Easy"])

def render_quiz_tab(agent: VideoResearchAgent, rag_manager: HybridRAGManager):
    """
    Renders the interactive Quiz Room interface.
    """
    st.subheader("🎯 Quiz Room")
    st.markdown("*Test your knowledge! Select a video, choose a difficulty level, and complete a 5-question multiple-choice quiz.*")
    
    indexed_videos = rag_manager.video_metadata_map
    if not indexed_videos:
        st.info("No videos currently indexed. Please index videos in the sidebar first!")
        return
        
    # Selection Form card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        selected_id = st.selectbox(
            "Select Quiz Topic (Video)",
            options=list(indexed_videos.keys()),
            format_func=lambda v_id: indexed_videos[v_id].get("title", f"Video {v_id}"),
            key="quiz_video_select"
        )
    with col_q2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="quiz_diff_select")
        
    generate_quiz = st.button("Generate Quiz", key="quiz_generate_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize state
    if "quiz" not in st.session_state:
        st.session_state.quiz = None
        
    # Check if we need to regenerate
    if generate_quiz or st.session_state.quiz is None or st.session_state.quiz.get("video_id") != selected_id or st.session_state.quiz.get("level") != difficulty:
        if generate_quiz or st.session_state.quiz is None:
            with st.spinner("Generating quiz questions from transcripts..."):
                meta = indexed_videos[selected_id]
                title = meta.get("title", "")
                
                # Fetch text content
                video_chunks = [c for c in rag_manager.chunks if c.video_id == selected_id]
                full_transcript = " ".join([c.text for c in video_chunks])
                
                if not agent._get_client() or "ySEx" in selected_id or not full_transcript:
                    questions = DEFAULT_QUIZ_QUESTIONS.get(difficulty, DEFAULT_QUIZ_QUESTIONS["Easy"])
                else:
                    questions = generate_quiz_from_llm(agent, title, full_transcript, difficulty)
                    
                st.session_state.quiz = {
                    "video_id": selected_id,
                    "level": difficulty,
                    "questions": questions,
                    "user_selections": {},
                    "submitted": False
                }
                st.rerun()

    # Render Active Quiz
    if st.session_state.quiz:
        quiz = st.session_state.quiz
        questions = quiz["questions"]
        
        st.write("---")
        st.markdown(f"### 📝 Quiz on *{indexed_videos[quiz['video_id']].get('title')}* ({quiz['level']} Level)")
        
        form_submitted = quiz["submitted"]
        
        # Display questions
        user_choices = {}
        for idx, q in enumerate(questions):
            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"**Question {idx + 1}:** {q['question']}")
            
            radio_key = f"quiz_q_{quiz['video_id']}_{quiz['level']}_{idx}"
            
            if form_submitted:
                selected_idx = quiz["user_selections"].get(idx)
                correct_idx = q["correct_answer_index"]
                
                st.radio(
                    "Options",
                    options=q["options"],
                    index=selected_idx,
                    key=radio_key + "_sub",
                    disabled=True
                )
                
                if selected_idx == correct_idx:
                    st.success("✅ **Correct!**")
                else:
                    st.error(f"❌ **Incorrect.** (You selected: {q['options'][selected_idx] if selected_idx is not None else 'None'})")
                    st.markdown(f"**Correct Answer:** {q['options'][correct_idx]}")
                    
                st.markdown(f"*Grounding Explanation:* {q['explanation']}")
            else:
                prev_val = quiz["user_selections"].get(idx)
                selected = st.radio(
                    "Options",
                    options=q["options"],
                    index=prev_val,
                    key=radio_key,
                    help="Select the correct answer"
                )
                if selected:
                    user_choices[idx] = q["options"].index(selected)
                    
            st.markdown('</div>', unsafe_allow_html=True)
            
        if not form_submitted:
            if st.button("Submit Quiz Answers", key="quiz_submit_btn"):
                quiz["user_selections"] = user_choices
                quiz["submitted"] = True
                st.rerun()
        else:
            score = sum(1 for idx, q in enumerate(questions) if quiz["user_selections"].get(idx) == q["correct_answer_index"])
            
            st.markdown(f'<div class="glass-card" style="text-align:center; border-color:#7c5cfc !important;">', unsafe_allow_html=True)
            st.markdown(f"## 🏆 Final Score: **{score} / {len(questions)}**")
            if score == len(questions):
                st.balloons()
                st.success("🎉 Perfect Score! Excellent job!")
            elif score >= 3:
                st.info("👍 Good job! You understand most of the concepts.")
            else:
                st.warning("📚 Consider reviewing the video transcript summaries and try again.")
                
            if st.button("Reset / New Quiz", key="quiz_reset_btn"):
                quiz["submitted"] = False
                quiz["user_selections"] = {}
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
