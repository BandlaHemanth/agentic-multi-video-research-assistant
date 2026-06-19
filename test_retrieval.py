"""
test_retrieval.py — CLI verification script for Phase 1.
Ingests a YouTube video URL, chunks the transcript, generates embeddings (mock or real),
adds them to the hybrid FAISS/BM25 index, and performs a test query showing timestamps.
ASCII-only output for maximum compatibility with all Windows console locales.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Ensure we load environment variables
load_dotenv()

from src.core.ingestion import ingest_video
from src.core.rag import HybridRAGManager
from config import GOOGLE_API_KEY

def format_time(seconds: float) -> str:
    """Helper to format seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def main():
    parser = argparse.ArgumentParser(description="Ingest a YouTube video and search its transcript.")
    parser.add_argument(
        "--url", 
        type=str, 
        default="https://www.youtube.com/watch?v=ySEx_BqVx8A", # What is an API video (short, fast, has manual transcript)
        help="YouTube video URL to ingest and search."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="what is an API", 
        help="Search query to test hybrid retrieval."
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-download and bypass ingestion cache."
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" PHASE 1 TESTING: VIDEO INGESTION & HYBRID RETRIEVAL INDEX ")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("NOTICE: GOOGLE_API_KEY is not set in your .env file.")
        print("   Running in DEMO MODE with local Cross-Encoder reranker and Mock (random) embeddings.")
        print("   Configure GOOGLE_API_KEY=your_key in .env to enable real semantic embeddings.")
        print("-" * 60)
    else:
        print("[OK] Real Semantic Embeddings Mode enabled (Gemini models/embedding-001).")
        print("-" * 60)
        
    # 1. Ingest Video
    print(f"\n[1/3] Ingesting video: {args.url}")
    try:
        video_data = ingest_video(args.url, force_refresh=args.force)
        print("[OK] Video Ingested Successfully!")
        print(f"  - Title: {video_data.metadata.title}")
        print(f"  - Author: {video_data.metadata.author}")
        print(f"  - Duration: {format_time(video_data.metadata.duration)}")
        print(f"  - View Count: {video_data.metadata.view_count:,}")
        print(f"  - Transcript Segments: {len(video_data.segments)}")
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        sys.exit(1)
        
    # 2. Add to Search Index
    print(f"\n[2/3] Chunking transcript and indexing into Hybrid RAG...")
    try:
        rag_manager = HybridRAGManager()
        # Clean index if force is enabled to test cleanly
        if args.force:
            print("  - Clearing index files and recreating...")
            if rag_manager.faiss_file.exists():
                os.remove(rag_manager.faiss_file)
            if rag_manager.metadata_file.exists():
                os.remove(rag_manager.metadata_file)
            rag_manager = HybridRAGManager()
            
        print(f"[INDEXING] id(rag_manager): {id(rag_manager)}")
        print(f"[INDEXING] len(rag_manager.chunks): {len(rag_manager.chunks)}")
        print(f"[INDEXING] len(rag_manager.video_metadata_map): {len(rag_manager.video_metadata_map)}")
        rag_manager.add_video(video_data)
        print(f"[INDEXING AFTER ADD] id(rag_manager): {id(rag_manager)}")
        print(f"[INDEXING AFTER ADD] len(rag_manager.chunks): {len(rag_manager.chunks)}")
        print(f"[INDEXING AFTER ADD] len(rag_manager.video_metadata_map): {len(rag_manager.video_metadata_map)}")
        print("[OK] Video Chunks Embedded and Indexed!")
        print(f"  - Total chunks in RAG index: {len(rag_manager.chunks)}")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")
        sys.exit(1)
        
    # 3. Perform Test Queries
    print(f"\n[3/3] Querying Index for: '{args.query}'")
    try:
        results = rag_manager.query_index(args.query, limit=3)
        print(f"[OK] Retrieval successful! Top {len(results)} matching chunks:")
        
        for i, chunk in enumerate(results, 1):
            meta = rag_manager.video_metadata_map.get(chunk.video_id, {})
            title = meta.get("title", "Unknown Title")
            
            print(f"\n  Match #{i} (Score: {chunk.score:.4f}):")
            print(f"    Video: {title} ({chunk.video_id})")
            print(f"    Timestamp: {format_time(chunk.start_time)} - {format_time(chunk.end_time)}")
            print(f"    Text snippet: \"{chunk.text[:200]}...\"")
    except Exception as e:
        print(f"[ERROR] Retrieval query failed: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print(" PHASE 1 VERIFICATION COMPLETED SUCCESSFULLY ")
    print("=" * 60)

if __name__ == "__main__":
    main()
