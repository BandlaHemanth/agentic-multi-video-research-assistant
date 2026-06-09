"""
test_generation.py — CLI verification script for Phase 2.
Loads the hybrid search index, runs a search query, generates a citations-embedded
response, executes a RAGAS evaluation, and logs the metrics history locally.
"""

import sys
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from src.core.rag import HybridRAGManager, format_time
from src.evaluation.evaluator import evaluate_rag_query, EVAL_LOG_FILE
from config import GOOGLE_API_KEY

def main():
    print("=" * 60)
    print(" PHASE 2 TESTING: RAG RESPONSE GENERATION & EVALUATION ")
    print("=" * 60)
    
    # 1. Load RAG Index
    print("\n[1/4] Loading RAG Search Index...")
    rag_manager = HybridRAGManager()
    if not rag_manager.chunks:
        print("INFO: Search index is empty. Please run Phase 1 test to index a video first.")
        print("      Or run: python test_retrieval.py")
        sys.exit(1)
        
    print(f"[OK] Index loaded successfully! Contains {len(rag_manager.chunks)} chunks.")
    
    # 2. Query and Generate Answer
    query = "what is an API and how does the hybrid search work"
    print(f"\n[2/4] Generating answer for query: '{query}'")
    
    try:
        answer, context_chunks = rag_manager.generate_answer(query, limit=3)
        print("[OK] Response generated successfully!")
        print("-" * 50)
        print("Generated Answer:")
        print(answer)
        print("-" * 50)
        
        # Print citations used
        print("Citations / Contexts retrieved:")
        for idx, chunk in enumerate(context_chunks, 1):
            meta = rag_manager.video_metadata_map.get(chunk.video_id, {})
            title = meta.get("title", "Video")
            print(f"  [{idx}] [{title} - {format_time(chunk.start_time)}]: \"{chunk.text[:100]}...\"")
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        sys.exit(1)
        
    # 3. Evaluate RAG Answer quality using Ragas
    print("\n[3/4] Evaluating retrieval/generation quality using Ragas...")
    context_texts = [c.text for c in context_chunks]
    
    # Optional ground truth for test query verification
    ground_truth = (
        "An API is an application programming interface that lets systems share data. "
        "A hybrid search combines dense vector search with sparse BM25 search."
    )
    
    try:
        scores = evaluate_rag_query(
            query=query,
            answer=answer,
            contexts=context_texts,
            ground_truth=ground_truth
        )
        print("[OK] Ragas evaluation completed!")
        print("-" * 50)
        print("Ragas Metric Scores:")
        for metric, score in scores.items():
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"  - {metric:20s}: {score_str}")
        print("-" * 50)
    except Exception as e:
        print(f"[ERROR] Evaluation execution failed: {e}")
        sys.exit(1)
        
    # 4. Confirm Log Persistence
    print("\n[4/4] Verifying local history logs...")
    if EVAL_LOG_FILE.exists():
        print(f"[OK] Evaluation results logged successfully to:")
        print(f"     {EVAL_LOG_FILE}")
    else:
        print("[ERROR] Evaluation log file was not created.")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print(" PHASE 2 VERIFICATION COMPLETED SUCCESSFULLY ")
    print("=" * 60)

if __name__ == "__main__":
    main()
