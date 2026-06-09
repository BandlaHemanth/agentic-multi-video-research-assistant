"""
rag.py — Chunking, embedding, indexing, hybrid retrieval, and reranking engine.
Integrates dense (FAISS with models/embedding-001) and sparse (BM25) search indexes
with normalized min-max hybrid blending and cross-encoder reranking.
Supports a mock embedding fallback if GOOGLE_API_KEY is not set.
"""

import os
import re
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import google.generativeai as genai

from config import (
    GOOGLE_API_KEY,
    GEMINI_LLM_MODEL,
    GEMINI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    FAISS_TOP_K,
    BM25_TOP_K,
    HYBRID_ALPHA,
    FINAL_CANDIDATES,
    RERANK_MODEL,
    RERANK_TOP_K,
    FAISS_INDEX_PATH
)
from src.core.models import TranscriptSegment, VideoData, RetrievalChunk

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure Gemini Generative AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY is not set. RAG will fall back to mock embeddings for testing.")

def format_time(seconds: float) -> str:
    """Helper to format seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def chunk_transcript(video_id: str, segments: List[TranscriptSegment]) -> List[RetrievalChunk]:
    """
    Groups consecutive transcript segments into overlapping, timestamp-tracked semantic chunks.
    
    Args:
        video_id (str): The video ID.
        segments (List[TranscriptSegment]): Individual transcript segments from ingestion.
        
    Returns:
        List[RetrievalChunk]: Overlapping chunks preserving time limits.
    """
    chunks: List[RetrievalChunk] = []
    if not segments:
        return chunks
        
    current_segments: List[TranscriptSegment] = []
    current_char_count = 0
    chunk_idx = 0
    
    for segment in segments:
        current_segments.append(segment)
        current_char_count += len(segment.text) + 1  # Add 1 for space separation
        
        while current_char_count >= CHUNK_SIZE:
            chunk_text = " ".join([s.text for s in current_segments])
            start_time = current_segments[0].start
            end_time = current_segments[-1].start + current_segments[-1].duration
            
            chunks.append(RetrievalChunk(
                chunk_id=f"{video_id}_chunk_{chunk_idx}",
                video_id=video_id,
                text=chunk_text,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                score=0.0
            ))
            chunk_idx += 1
            
            # Guard against single massive segment causing infinite loop
            if len(current_segments) <= 1:
                current_segments = []
                current_char_count = 0
                break
                
            # Pop the first element from current window to enforce sliding overlap
            removed = current_segments.pop(0)
            current_char_count -= len(removed.text) + 1
            
    # Flush remaining segments
    if current_segments:
        chunk_text = " ".join([s.text for s in current_segments])
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            start_time = current_segments[0].start
            end_time = current_segments[-1].start + current_segments[-1].duration
            chunks.append(RetrievalChunk(
                chunk_id=f"{video_id}_chunk_{chunk_idx}",
                video_id=video_id,
                text=chunk_text,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                score=0.0
            ))
            
    logger.info(f"Chunked video {video_id} into {len(chunks)} chunks.")
    return chunks

class HybridRAGManager:
    """
    Manages vector storage (FAISS) and lexical storage (BM25) to perform
    hybrid search and reranking on timestamped transcript chunks.
    """
    def __init__(self, index_path: Path = FAISS_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Paths for specific persistence files
        self.faiss_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.json"
        
        # Dense storage structures
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        
        # In-memory datasets
        self.chunks: List[RetrievalChunk] = []
        self.video_metadata_map: Dict[str, Dict] = {}  # video_id -> metadata_dict
        
        # Sparse BM25 index
        self.bm25: Optional[BM25Okapi] = None
        
        # Lazy load cross-encoder
        self._reranker: Optional[CrossEncoder] = None
        
        # Load index files if they exist
        self.load_index()

    @property
    def reranker(self) -> CrossEncoder:
        """Lazily load the cross-encoder model to save overhead when unused."""
        if self._reranker is None:
            logger.info(f"Loading CrossEncoder reranker model: {RERANK_MODEL}")
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def _tokenize(self, text: str) -> List[str]:
        """Simple, fast tokenization suitable for BM25 matching."""
        return re.findall(r'\w+', text.lower())

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Retrieves dense embeddings from Gemini API in batches of 100.
        Falls back to random mock unit vectors if GOOGLE_API_KEY is not set.
        
        Args:
            texts (List[str]): Chunks to embed.
            
        Returns:
            np.ndarray: Vector representations (normed).
        """
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not configured. Generating mock (random) embeddings for testing.")
            arr = np.random.randn(len(texts), EMBEDDING_DIM).astype(np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
            
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            all_embeddings.extend(response['embedding'])
            
        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def add_video(self, video_data: VideoData) -> None:
        """
        Processes a video: chunks its transcript, generates embeddings,
        updates FAISS/BM25 indexes, and saves files.
        
        Args:
            video_data (VideoData): The video's parsed transcripts and metadata.
        """
        video_id = video_data.metadata.video_id
        
        # 1. Chunk transcript
        new_chunks = chunk_transcript(video_id, video_data.segments)
        if not new_chunks:
            logger.warning(f"No chunks created for video {video_id}.")
            return
            
        # Track metadata mapping
        self.video_metadata_map[video_id] = video_data.metadata.__dict__
        
        # 2. Get embeddings for new chunks
        chunk_texts = [c.text for c in new_chunks]
        embeddings = self._get_embeddings(chunk_texts)
        
        # 3. Add to FAISS index
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            
        self.faiss_index.add(embeddings)
        
        # 4. Append to local chunks database
        self.chunks.extend(new_chunks)
        
        # 5. Rebuild BM25 Lexical Index
        tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 6. Persist database & index
        self.save_index()

    def remove_video(self, video_id: str) -> None:
        """
        Removes a video and its transcript chunks from the database,
        rebuilds the dense and sparse indexes from the remaining chunks,
        and persists the changes.
        """
        if video_id not in self.video_metadata_map:
            logger.warning(f"Video {video_id} not found in index metadata map.")
            return
            
        # 1. Remove metadata entry
        del self.video_metadata_map[video_id]
        
        # 2. Filter remaining chunks
        self.chunks = [c for c in self.chunks if c.video_id != video_id]
        
        # 3. Handle empty index case
        if not self.chunks:
            self.faiss_index = None
            self.bm25 = None
            # Delete files on disk
            if self.faiss_file.exists():
                self.faiss_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info(f"Removed video {video_id}. Index is now empty.")
            return
            
        # 4. Rebuild FAISS index
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self._get_embeddings(chunk_texts)
        self.faiss_index.add(embeddings)
        
        # 5. Rebuild BM25
        tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 6. Save changes
        self.save_index()
        logger.info(f"Successfully removed video {video_id} and rebuilt indexes.")

    def save_index(self) -> None:
        """Saves current FAISS index and chunks metadata to disk."""
        if self.faiss_index is None or not self.chunks:
            logger.warning("No data to save in the indices.")
            return
            
        try:
            # Write FAISS
            faiss.write_index(self.faiss_index, str(self.faiss_file))
            
            # Write Chunks Metadata & Video Metadata Maps
            metadata = {
                'video_metadata': self.video_metadata_map,
                'chunks': [c.__dict__ for c in self.chunks]
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            logger.info("Successfully persisted FAISS and metadata index to disk.")
        except Exception as e:
            logger.error(f"Error saving RAG index: {e}")

    def load_index(self) -> None:
        """Loads FAISS index and chunk metadata from disk if available."""
        if not self.faiss_file.exists() or not self.metadata_file.exists():
            logger.info("No existing search indices found. Initializing empty RAG manager.")
            return
            
        try:
            # Read FAISS
            self.faiss_index = faiss.read_index(str(self.faiss_file))
            
            # Read Chunks Metadata & Video Metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            self.video_metadata_map = metadata.get('video_metadata', {})
            
            self.chunks = []
            for c_data in metadata.get('chunks', []):
                self.chunks.append(RetrievalChunk(
                    chunk_id=c_data['chunk_id'],
                    video_id=c_data['video_id'],
                    text=c_data['text'],
                    start_time=c_data['start_time'],
                    end_time=c_data['end_time'],
                    score=c_data.get('score', 0.0)
                ))
                
            # Rebuild BM25
            tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Successfully loaded search index with {len(self.chunks)} chunks from {len(self.video_metadata_map)} videos.")
        except Exception as e:
            logger.error(f"Failed to load search indexes: {e}. Reinitializing empty.")
            self.faiss_index = None
            self.chunks = []
            self.video_metadata_map = {}
            self.bm25 = None

    def dense_search(self, query: str, k: int = FAISS_TOP_K) -> List[Tuple[int, float]]:
        """
        Searches the dense FAISS vector index.
        
        Args:
            query (str): Search query.
            k (int): Number of nearest neighbors.
            
        Returns:
            List[Tuple[int, float]]: List of tuples (chunk_index, similarity_score).
        """
        if self.faiss_index is None or not self.chunks:
            return []
            
        # Get query embedding
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not configured. Generating mock (random) query embedding.")
            query_emb = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
        else:
            query_response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            query_emb = np.array([query_response['embedding']], dtype=np.float32)
        
        norm = np.linalg.norm(query_emb[0])
        if norm > 0:
            query_emb[0] = query_emb[0] / norm
            
        k = min(k, len(self.chunks))
        scores, indices = self.faiss_index.search(query_emb, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        return results

    def sparse_search(self, query: str, k: int = BM25_TOP_K) -> List[Tuple[int, float]]:
        """
        Searches the sparse BM25 lexical index.
        
        Args:
            query (str): Search query.
            k (int): Number of candidates.
            
        Returns:
            List[Tuple[int, float]]: List of tuples (chunk_index, bm25_score).
        """
        if self.bm25 is None or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((int(idx), score))
        return results

    def hybrid_search(self, query: str, alpha: float = HYBRID_ALPHA, top_k: int = FINAL_CANDIDATES) -> List[RetrievalChunk]:
        """
        Combines dense (FAISS) and sparse (BM25) retrieval using normalized min-max score blending.
        
        Args:
            query (str): The search query.
            alpha (float): Scaling factor.
            top_k (int): Number of candidates to select.
            
        Returns:
            List[RetrievalChunk]: Blended, score-updated retrieval candidates.
        """
        dense_res = self.dense_search(query, k=max(FAISS_TOP_K, top_k))
        sparse_res = self.sparse_search(query, k=max(BM25_TOP_K, top_k))
        
        if not dense_res and not sparse_res:
            return []
            
        raw_dense_scores = {idx: score for idx, score in dense_res}
        raw_sparse_scores = {idx: score for idx, score in sparse_res}
            
        def normalize_scores(res_list: List[Tuple[int, float]]) -> Dict[int, float]:
            if not res_list:
                return {}
            indices, scores = zip(*res_list)
            max_s = max(scores)
            min_s = min(scores)
            denom = max_s - min_s
            
            normalized = {}
            for idx, score in res_list:
                normalized[idx] = (score - min_s) / denom if denom > 0.0 else 1.0
            return normalized
            
        norm_dense = normalize_scores(dense_res)
        norm_sparse = normalize_scores(sparse_res)
        
        merged_scores: Dict[int, float] = {}
        all_indices = set(norm_dense.keys()).union(set(norm_sparse.keys()))
        
        for idx in all_indices:
            d_score = norm_dense.get(idx, 0.0)
            s_score = norm_sparse.get(idx, 0.0)
            merged_scores[idx] = alpha * d_score + (1.0 - alpha) * s_score
            
        sorted_indices = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        
        blended_chunks = []
        for idx, score in sorted_indices:
            orig_chunk = self.chunks[idx]
            chunk_copy = RetrievalChunk(
                chunk_id=orig_chunk.chunk_id,
                video_id=orig_chunk.video_id,
                text=orig_chunk.text,
                start_time=orig_chunk.start_time,
                end_time=orig_chunk.end_time,
                score=score
            )
            # Attach raw metrics dynamically for debug visibility
            chunk_copy.dense_score = raw_dense_scores.get(idx, 0.0)
            chunk_copy.sparse_score = raw_sparse_scores.get(idx, 0.0)
            chunk_copy.hybrid_score = score
            blended_chunks.append(chunk_copy)
            
        return blended_chunks

    def rerank(self, query: str, chunks: List[RetrievalChunk], top_k: int = RERANK_TOP_K) -> List[RetrievalChunk]:
        """
        Reranks a list of candidate chunks using a cross-encoder model.
        
        Args:
            query (str): The search query.
            chunks (List[RetrievalChunk]): Candidate chunks to rerank.
            top_k (int): Number of final chunks to return.
            
        Returns:
            List[RetrievalChunk]: Top reranked, score-updated chunks.
        """
        if not chunks:
            return []
            
        pairs = [[query, chunk.text] for chunk in chunks]
        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.error(f"Error executing reranking: {e}. Returning original ranking order.")
            # Set default rerank scores
            for chunk in chunks:
                chunk.rerank_score = 0.0
            return chunks[:top_k]
            
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            chunk.rerank_score = float(score)
            
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[:top_k]

    def query_index(self, query: str, alpha: float = HYBRID_ALPHA, limit: int = RERANK_TOP_K) -> List[RetrievalChunk]:
        """
        High-level search interface: performs hybrid retrieval followed by cross-encoder reranking.
        
        Args:
            query (str): User's question.
            alpha (float): dense vs sparse fusion weight.
            limit (int): Final output chunk count.
            
        Returns:
            List[RetrievalChunk]: Reranked top-k matching transcript chunks.
        """
        import time
        start_time = time.time()
        
        # 1. Hybrid Search
        hybrid_start = time.time()
        hybrid_candidates = self.hybrid_search(query, alpha=alpha, top_k=FINAL_CANDIDATES)
        hybrid_latency = (time.time() - hybrid_start) * 1000
        
        # 2. Reranking
        rerank_start = time.time()
        reranked = self.rerank(query, hybrid_candidates, top_k=limit)
        rerank_latency = (time.time() - rerank_start) * 1000
        
        total_latency = (time.time() - start_time) * 1000
        
        # Attach latency metrics to chunks dynamically
        for chunk in reranked:
            chunk.dense_latency_ms = hybrid_latency
            chunk.rerank_latency_ms = rerank_latency
            chunk.total_latency_ms = total_latency
            
        return reranked

    def generate_answer(self, query: str, alpha: float = HYBRID_ALPHA, limit: int = RERANK_TOP_K) -> Tuple[str, List[RetrievalChunk]]:
        """
        Retrieves matching chunks and generates an answer using Gemini 2.5 Flash.
        
        Args:
            query (str): The search query.
            alpha (float): Dense/sparse hybrid blend factor.
            limit (int): Number of chunks to retrieve for context.
            
        Returns:
            Tuple[str, List[RetrievalChunk]]: (Generated answer, List of retrieved chunks used).
        """
        # 1. Retrieve top reranked chunks
        chunks = self.query_index(query, alpha=alpha, limit=limit)
        if not chunks:
            return "No relevant video context could be found to answer this question.", []
            
        # 2. Build prompt context
        context_blocks = []
        for c in chunks:
            meta = self.video_metadata_map.get(c.video_id, {})
            title = meta.get("title", "Unknown Title")
            start_str = format_time(c.start_time)
            context_blocks.append(
                f"Source Video: {title} (ID: {c.video_id})\n"
                f"Timestamp: {start_str} - {format_time(c.end_time)}\n"
                f"Transcript snippet: {c.text}\n"
                f"---"
            )
        context_str = "\n".join(context_blocks)
        
        # 3. Generate response using Gemini 2.5 Flash
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is not set. Generating mock/demo answer.")
            summary_points = []
            for c in chunks:
                meta = self.video_metadata_map.get(c.video_id, {})
                title = meta.get("title", "Video")
                summary_points.append(
                    f"According to [{title} - {format_time(c.start_time)}], the system "
                    f"retrieved this text snippet: '{c.text[:100]}...'"
                )
            answer = (
                "[DEMO MODE] This is a mock response synthesized from retrieved chunks:\n\n" +
                "\n".join(summary_points) +
                "\n\nTo generate actual responses, please configure GOOGLE_API_KEY in your .env file."
            )
            return answer, chunks
            
        # Real LLM Call
        prompt = f"""You are an expert research assistant. Answer the following question using ONLY the provided video transcript context (which may be in different languages).
Always respond in the language of the user's question (default: English).
For each key claim, fact, or quote, you MUST cite the source using the exact format: [Video Title - MM:SS].
Ensure that the timestamp (MM:SS) matches the start time of the context snippet.
Keep your answer factual and direct. If the context does not contain the answer, say "I do not know based on the provided video contexts."

Context:
{context_str}

Question:
{query}

Answer:"""
        
        try:
            model = genai.GenerativeModel(GEMINI_LLM_MODEL)
            response = model.generate_content(prompt)
            return response.text, chunks
        except Exception as e:
            logger.error(f"Error generating answer via Gemini: {e}")
            return f"Error occurred during response generation: {e}", chunks

if __name__ == "__main__":
    manager = HybridRAGManager()
    print(f"RAG Manager loaded with {len(manager.chunks)} chunks.")
