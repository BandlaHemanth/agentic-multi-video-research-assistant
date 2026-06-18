"""
evaluator.py — RAGAS-based retrieval and answer evaluation engine.
Configures RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision/Recall)
using Gemini 2.5 Flash, stubs VertexAI imports for compatibility, logs evaluations,
and provides a fallback mode for demo runs without API keys.
"""

import os
import sys
import types
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# VERTEXAI COMPATIBILITY STUB FOR RAGAS
# Ragas 0.4.3 imports langchain_community.chat_models.vertexai ChatVertexAI
# dynamically, which is deprecated/removed in newer LangChain versions.
# We stub this package in sys.modules to prevent import crashes.
# ──────────────────────────────────────────────────────────────────────────────
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    # Stub langchain_community.chat_models
    chat_models = types.ModuleType("langchain_community.chat_models")
    sys.modules["langchain_community.chat_models"] = chat_models
    
    # Stub vertexai
    vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    class DummyChatVertexAI:
        pass
    vertexai.ChatVertexAI = DummyChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai

import google.genai as google_genai

from config import GOOGLE_API_KEY, RAGAS_EVAL_LLM_MODEL, LOG_DIR

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EVAL_LOG_FILE = LOG_DIR / "evaluation_history.json"

def evaluate_rag_query(
    query: str, 
    answer: str, 
    contexts: List[str], 
    ground_truth: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """
    Evaluates a single RAG query's quality using RAGAS metrics.
    If GOOGLE_API_KEY is not set, returns a mock evaluation report for demo/testing.
    
    Args:
        query (str): User query/question.
        answer (str): Generated LLM answer.
        contexts (List[str]): Contexts retrieved.
        ground_truth (Optional[str]): Correct/ideal reference answer (required for Context Recall).
        
    Returns:
        Dict[str, Optional[float]]: Evaluated scores for faithfulness, relevancy, etc.
    """
    scores: Dict[str, Optional[float]] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None
    }
    
    # 1. Run in Mock/Demo Mode if key is missing
    active_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not active_key:
        logger.warning("GOOGLE_API_KEY not configured. Generating mock evaluation scores for testing.")
        scores.update({
            "faithfulness": 0.95 if "API" in answer or "RAG" in answer else 0.85,
            "answer_relevancy": 0.90 if len(query) > 5 else 0.70,
            "context_precision": 0.88,
            "context_recall": 0.92 if ground_truth else None
        })
        log_evaluation(query, answer, contexts, scores, ground_truth)
        return scores
        
    # 2. Run real Ragas evaluation
    try:
        # Lazy load RAGAS modules to prevent slow app startup times
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.llms import llm_factory
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # Define dataset entry
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts]
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            
        dataset = Dataset.from_dict(data)
        
        # Configure Gemini Evaluator LLM using modern Ragas InstructorLLM (llm_factory)
        google_client = google_genai.Client(api_key=active_key)
        ragas_llm = llm_factory(
            model=RAGAS_EVAL_LLM_MODEL,  # "gemini-2.5-flash"
            provider="google",
            client=google_client
        )
        
        # Instantiate langchain embeddings and wrap it for Ragas
        langchain_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=active_key
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        
        # Instantiate metric classes as required by Ragas 0.4.3 (requires llm/embeddings arguments)
        active_metrics = [
            Faithfulness(llm=ragas_llm), 
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings), 
            ContextPrecision(llm=ragas_llm)
        ]
        if ground_truth:
            active_metrics.append(ContextRecall(llm=ragas_llm))
            
        logger.info("Executing RAGAS evaluation...")
        result = evaluate(
            dataset=dataset,
            metrics=active_metrics
        )
        
        # Extract and format scores from the RAGAS EvaluationResult object
        import math
        def safe_float(val) -> float:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return 0.0
            try:
                f_val = float(val)
                return 0.0 if math.isnan(f_val) else f_val
            except (ValueError, TypeError):
                return 0.0

        repr_dict = getattr(result, "_repr_dict", {})
        scores["faithfulness"] = safe_float(repr_dict.get("faithfulness"))
        scores["answer_relevancy"] = safe_float(repr_dict.get("answer_relevancy"))
        scores["context_precision"] = safe_float(repr_dict.get("context_precision"))
        if ground_truth:
            scores["context_recall"] = safe_float(repr_dict.get("context_recall"))
            
        logger.info(f"RAGAS evaluation success. Scores: {scores}")
    except Exception as e:
        logger.error(f"Failed to execute RAGAS evaluation: {e}")
        scores.update({
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0 if ground_truth else None
        })
        
    # 3. Log evaluation results to disk
    log_evaluation(query, answer, contexts, scores, ground_truth)
    return scores

def log_evaluation(
    query: str, 
    answer: str, 
    contexts: List[str], 
    scores: Dict[str, Optional[float]], 
    ground_truth: Optional[str] = None
) -> None:
    """
    Appends an evaluation result into the local JSON file log.
    """
    history = []
    if EVAL_LOG_FILE.exists():
        try:
            with open(EVAL_LOG_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluation history: {e}. Resetting history logs.")
            
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "scores": scores
    }
    history.append(new_entry)
    
    try:
        with open(EVAL_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation results persisted to {EVAL_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save evaluation log: {e}")

def get_evaluation_history() -> List[Dict]:
    """Loads all historically completed RAGAS evaluations from disk."""
    if not EVAL_LOG_FILE.exists():
        return []
    try:
        with open(EVAL_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load evaluation history: {e}")
        return []

if __name__ == "__main__":
    print("Evaluator module initialized.")
