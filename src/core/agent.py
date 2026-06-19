"""
agent.py — Agentic Core for the Multi-Video Research Assistant.
Defines execution tracing dataclasses and implements the VideoResearchAgent
which orchestrates a custom ReAct loop with tools (search, details, summarize)
using the native tool-calling features of Gemini 2.5 Flash, backed by rate-limit retries.
"""

import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import google.genai as google_genai
from google.genai import types

# Import ONLY the model name constant — never import GOOGLE_API_KEY
from config import GEMINI_LLM_MODEL
from src.core.models import RetrievalChunk
from src.core.rag import HybridRAGManager, format_time

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AgentTraceStep:
    """Tracks a single step in the agent's reasoning and action sequence."""
    step_index: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentExecutionResult:
    """Stores the final result of an agent run along with verification context."""
    answer: str
    retrieved_chunks: List[RetrievalChunk] = field(default_factory=list)
    trace_steps: List[AgentTraceStep] = field(default_factory=list)
    is_fallback: bool = False


class VideoResearchAgent:
    """
    An agentic controller that runs a custom ReAct (Reasoning and Action) loop.
    Exposes search, metadata lookup, and summarization tools to Gemini 2.5 Flash,
    captures execution trace steps, and formats final cited answers.

    IMPORTANT: No Gemini client is cached at __init__ time.
    Every Gemini operation calls _get_client() which reads os.environ["GOOGLE_API_KEY"]
    fresh at call-time, so it always picks up the key resolved by app.py from
    Streamlit Secrets or session overrides — regardless of when this object was created.
    """

    def __init__(self, rag_manager: Optional[HybridRAGManager] = None):
        self.rag_manager = rag_manager or HybridRAGManager()
        print(f"[AGENT.__INIT__] id(self.rag_manager): {id(self.rag_manager)}")
        print(f"[AGENT.__INIT__] len(self.rag_manager.chunks): {len(self.rag_manager.chunks)}")
        print(f"[AGENT.__INIT__] len(self.rag_manager.video_metadata_map): {len(self.rag_manager.video_metadata_map)}")
        self.last_retrieved_chunks: List[RetrievalChunk] = []
        self.trace_steps: List[AgentTraceStep] = []
        # NOTE: No self.client here. Use _get_client() at call-time.
        logger.info("VideoResearchAgent initialized. Client will be resolved at call-time via _get_client().")

    # ──────────────────────────────────────────────────────────────────
    # RUNTIME CLIENT — reads os.environ at every call, never stale
    # ──────────────────────────────────────────────────────────────────
    def _get_client(self) -> Optional[google_genai.Client]:
        """
        Returns a fresh google.genai.Client using the API key currently in
        os.environ["GOOGLE_API_KEY"]. Returns None if the key is missing.

        This is called at the start of every Gemini operation so that it always
        uses the key that app.py resolved from Streamlit Secrets, environment
        variables, or user session overrides — not a stale import-time constant.
        """
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key:
            logger.warning("[_get_client] GOOGLE_API_KEY is not set in os.environ. Will run in Demo/Mock mode.")
            return None
        return google_genai.Client(api_key=key)

    # ──────────────────────────────────────────────────────────────────
    # RETRY WRAPPER
    # ──────────────────────────────────────────────────────────────────
    def call_with_retry(self, fn, *args, **kwargs) -> Any:
        """
        Executes a Gemini API function, automatically retrying with exponential backoff
        and reading wait times directly from 429/503 error responses.
        
        CRITICAL: If the error is HTTP 429 / RESOURCE_EXHAUSTED / quota exceeded / rate limited / 503 / UNAVAILABLE,
        we raise it immediately so the agent falls back to local summary without retrying or waiting.
        """
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                # Log Gemini request start timestamp
                print(f"[TIMING] Gemini request start at {time.time():.3f}", flush=True)
                logger.info(f"[TIMING] Gemini request start at {time.time():.3f}")
                return fn(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                # Log Gemini exception raised timestamp
                print(f"[TIMING] Gemini exception raised at {time.time():.3f} — {err_str}", flush=True)
                logger.info(f"[TIMING] Gemini exception raised at {time.time():.3f} — {err_str}")
                
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower() or "rate limit" in err_str.lower()
                is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str or "unavailable" in err_str.lower()

                if is_rate_limit or is_unavailable:
                    # Immediately raise the exception to switch to fallback, no retry
                    raise e

                if attempt < max_attempts:
                    # Parse retry delay
                    wait_time = 5
                    logger.warning(
                        f"[Attempt {attempt}/{max_attempts}] Gemini API transient error. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e


    # ──────────────────────────────────────────────────────────────────
    # TOOL IMPLEMENTATIONS
    # ──────────────────────────────────────────────────────────────────
    def _search_videos(self, query: str) -> str:
        """Helper executing RAG retrieval and accumulating context chunks."""
        logger.info(f"Tool Action: search_videos with query: '{query}'")
        
        import sys, traceback
        print(f"[{time.time():.3f}] [RUN] Step 2 hybrid search start")
        try:
            hybrid_candidates = self.rag_manager.hybrid_search(query)
        except Exception as e:
            tb = sys.exc_info()[2]
            print(f"[RUN] Exception in hybrid search:")
            print(f"  Filename: {tb.tb_frame.f_code.co_filename}")
            print(f"  Line: {tb.tb_lineno}")
            print(f"  Class: {type(e).__name__}")
            print(f"  Message: {e}")
            traceback.print_exc()
            raise e
        print(f"[{time.time():.3f}] [RUN] Step 3 hybrid search complete")
        
        print(f"[{time.time():.3f}] [RUN] Step 4 reranker start")
        try:
            chunks = self.rag_manager.rerank(query, hybrid_candidates)
        except Exception as e:
            tb = sys.exc_info()[2]
            print(f"[RUN] Exception in reranker:")
            print(f"  Filename: {tb.tb_frame.f_code.co_filename}")
            print(f"  Line: {tb.tb_lineno}")
            print(f"  Class: {type(e).__name__}")
            print(f"  Message: {e}")
            traceback.print_exc()
            raise e
        print(f"[{time.time():.3f}] [RUN] Step 5 reranker complete")
        
        self.last_retrieved_chunks.extend(chunks)

        if not chunks:
            return "Observation: No relevant video transcript segments found for this search."

        formatted_results = []
        for idx, chunk in enumerate(chunks, 1):
            meta = self.rag_manager.video_metadata_map.get(chunk.video_id, {})
            title = meta.get("title", "Unknown Title")
            formatted_results.append(
                f"Result {idx}:\n"
                f"  Video ID: {chunk.video_id}\n"
                f"  Title: {title}\n"
                f"  Timestamp: {format_time(chunk.start_time)} - {format_time(chunk.end_time)}\n"
                f"  Text: {chunk.text}\n"
            )
        return "Observation:\n" + "\n".join(formatted_results)

    def _get_video_details(self, video_id: str) -> str:
        """Helper retrieving video metadata."""
        logger.info(f"Tool Action: get_video_details for video_id: '{video_id}'")
        meta = self.rag_manager.video_metadata_map.get(video_id)
        if not meta:
            return f"Observation: Video with ID '{video_id}' not found in the search index database."

        details = (
            f"Observation:\n"
            f"  Video ID: {video_id}\n"
            f"  Title: {meta.get('title', 'Unknown')}\n"
            f"  Uploader: {meta.get('uploader', 'Unknown')}\n"
            f"  Duration: {format_time(meta.get('duration', 0.0))} (seconds: {meta.get('duration', 0)})\n"
            f"  Views: {meta.get('view_count', 'Unknown')}\n"
            f"  Publish Date: {meta.get('upload_date', 'Unknown')}\n"
        )
        return details

    def _summarize_video(self, video_id: str) -> str:
        """Helper generating transcript summaries using Gemini."""
        logger.info(f"Tool Action: summarize_video for video_id: '{video_id}'")
        meta = self.rag_manager.video_metadata_map.get(video_id)
        if not meta:
            return f"Observation: Video with ID '{video_id}' not found in database to summarize."

        # Collect and combine text chunks belonging to this video
        video_chunks = [c for c in self.rag_manager.chunks if c.video_id == video_id]
        if not video_chunks:
            return f"Observation: No transcript content found for video '{video_id}'."

        # Sort chunks chronologically by start_time
        video_chunks = sorted(video_chunks, key=lambda c: c.start_time)
        full_transcript = " ".join([c.text for c in video_chunks])

        # Get a fresh client at call-time
        client = self._get_client()
        if not client:
            return self._generate_local_fallback_summary(video_id, video_chunks)

        prompt = (
            f"Please generate a concise, professional summary of the following video transcript. "
            f"Focus on the main concepts discussed and list key takeaways.\n\n"
            f"Video Title: {meta.get('title', 'Unknown')}\n"
            f"Transcript:\n{full_transcript}\n\n"
            f"Summary:"
        )

        try:
            response = self.call_with_retry(
                client.models.generate_content,
                model=GEMINI_LLM_MODEL,
                contents=prompt
            )
            return f"Observation:\n{response.text}"
        except Exception as e:
            return f"Observation: Error summarizing video transcript via Gemini: {e}"

    # ──────────────────────────────────────────────────────────────────
    # DETAILED LOCAL DETERMINISTIC FALLBACK SYSTEM (PRODUCTION-GRADE)
    # ──────────────────────────────────────────────────────────────────
    def _generate_local_fallback_summary(self, video_id: str, chunks: List[RetrievalChunk]) -> str:
        """Generates a structured, deterministic local summary from retrieved transcript chunks."""
        meta = self.rag_manager.video_metadata_map.get(video_id, {})
        title = meta.get("title", "Video")
        uploader = meta.get("author", meta.get("uploader", "Unknown"))
        duration_str = format_time(meta.get("duration", 0))

        # Filter chunks for this video
        video_chunks = [c for c in chunks if c.video_id == video_id]
        if not video_chunks:
            # Fallback: get top chronological chunks for this video
            video_chunks = [c for c in self.rag_manager.chunks if c.video_id == video_id][:5]

        # Sort chunks chronologically by start_time
        video_chunks = sorted(video_chunks, key=lambda c: c.start_time)

        # Build the summary
        summary_lines = []
        summary_lines.append(f"### 🎥 Local Summary: {title}")
        summary_lines.append(f"**Uploader:** {uploader} | **Duration:** {duration_str}")
        summary_lines.append("")
        
        # Extract first sentence/part of the first chronological chunk or synthesize a topic line
        first_chunk_text = video_chunks[0].text if video_chunks else ""
        first_sentence = ""
        if first_chunk_text:
            sentences = re.split(r'\.\s+', first_chunk_text)
            if sentences:
                first_sentence = sentences[0].strip()
        
        summary_lines.append("#### 📌 Overall Topic")
        if first_sentence:
            summary_lines.append(f"Based on the transcript segments, this video discusses: *\"{first_sentence}...\"*")
        else:
            summary_lines.append(f"This video is an indexed presentation titled \"{title}\" uploaded by {uploader}.")
        summary_lines.append("")

        # Main Ideas & Important Details
        summary_lines.append("#### 🔑 Main Ideas & Important Details")
        for chunk in video_chunks:
            start_str = format_time(chunk.start_time)
            chunk_sentences = re.split(r'\.\s+', chunk.text)
            cleaned_sentences = [s.strip() for s in chunk_sentences if len(s.strip()) > 10]
            
            if cleaned_sentences:
                snippet = ". ".join(cleaned_sentences[:2])
                if not snippet.endswith('.'):
                    snippet += '.'
                summary_lines.append(f"- **[{start_str}]** {snippet}")
            else:
                summary_lines.append(f"- **[{start_str}]** {chunk.text[:150]}...")
        
        summary_lines.append("")
        summary_lines.append("---")
        summary_lines.append("*⚠️ **Note:** Gemini API is temporarily unavailable or has reached its quota. Displaying a local retrieval-based summary instead.*")

        return "\n".join(summary_lines)

    def _generate_local_fallback_qa(self, query: str, chunks: List[RetrievalChunk]) -> str:
        """Generates a structured, deterministic local answer for a general query from retrieved chunks."""
        summary_lines = []
        summary_lines.append("### 🔍 Local Retrieval-based Answer")
        summary_lines.append(f"**Query:** \"{query}\"")
        summary_lines.append("")
        summary_lines.append("#### 📑 Relevant Transcript Chunks Found:")
        
        if not chunks:
            summary_lines.append("No relevant transcript segments were found in the index to answer this query.")
        else:
            for i, chunk in enumerate(chunks[:3], 1):
                meta = self.rag_manager.video_metadata_map.get(chunk.video_id, {})
                title = meta.get("title", "Video")
                start_str = format_time(chunk.start_time)
                summary_lines.append(f"**Source {i}:** [{title} - {start_str}]")
                summary_lines.append(f"> {chunk.text.strip()}")
                summary_lines.append("")
                
        summary_lines.append("---")
        summary_lines.append("*⚠️ **Note:** Gemini API is temporarily unavailable or has reached its quota. Displaying a local retrieval-based summary instead.*")
        
        return "\n".join(summary_lines)

    def _run_mock_fallback(self, user_query: str) -> AgentExecutionResult:
        """
        Runs a local, deterministic fallback summary or QA generator grounded in retrieved context
        when the Gemini API is rate-limited (HTTP 429), unavailable, or missing.
        """
        fallback_start = time.time()

        video_ids = list(self.rag_manager.video_metadata_map.keys())
        
        if not video_ids:
            answer = (
                "No videos are currently indexed. Please index a video first.\n\n"
                "*(Note: Gemini API is temporarily unavailable or has reached its quota.)*"
            )
            return AgentExecutionResult(answer=answer, retrieved_chunks=[], trace_steps=[], is_fallback=True)

        # 1. Hybrid Retrieval Timing
        retrieval_start = time.time()
        try:
            # Limit to 10 candidates before cross-encoding to optimize performance
            hybrid_candidates = self.rag_manager.hybrid_search(user_query, top_k=10)
        except Exception as e:
            logger.error(f"Fallback hybrid retrieval failed: {e}")
            hybrid_candidates = []
        retrieval_latency = time.time() - retrieval_start
        print(f"[TIMING] Local Hybrid Retrieval took {retrieval_latency:.4f} seconds.")

        # 2. Cross-Encoder Reranking Timing
        rerank_start = time.time()
        try:
            retrieved = self.rag_manager.rerank(user_query, hybrid_candidates, top_k=5)
        except Exception as e:
            logger.error(f"Fallback reranking failed: {e}")
            retrieved = hybrid_candidates[:5]
        rerank_latency = time.time() - rerank_start
        print(f"[TIMING] Local Cross-Encoder Reranking took {rerank_latency:.4f} seconds.")

        self.last_retrieved_chunks = retrieved

        # 3. Summary/QA Generation Timing
        gen_start = time.time()
        normalized_q = re.sub(r'[^\w\s]', '', user_query.lower()).strip()
        is_summary_req = any(x in normalized_q for x in ["summarize", "summary", "overview", "what is this video about"])

        if is_summary_req:
            selected_video_id = None
            if len(video_ids) == 1:
                selected_video_id = video_ids[0]
            else:
                for v_id in video_ids:
                    if v_id in user_query:
                        selected_video_id = v_id
                        break
                if not selected_video_id:
                    for v_id, meta in self.rag_manager.video_metadata_map.items():
                        title = meta.get("title", "")
                        if title and title.lower() in user_query.lower():
                            selected_video_id = v_id
                            break
                if not selected_video_id and retrieved:
                    counts = {}
                    for c in retrieved:
                        counts[c.video_id] = counts.get(c.video_id, 0) + 1
                    selected_video_id = max(counts, key=counts.get)
            
            if selected_video_id:
                answer = self._generate_local_fallback_summary(selected_video_id, retrieved)
            else:
                clarification_msg = (
                    "Multiple videos are currently indexed. Please clarify which video you would like to summarize:\n\n"
                )
                for v_id in video_ids:
                    meta = self.rag_manager.video_metadata_map.get(v_id, {})
                    title = meta.get("title", "Unknown Title")
                    clarification_msg += f"- **{title}** (Video ID: `{v_id}`)\n"
                clarification_msg += "\n*⚠️ Gemini API is temporarily unavailable or has reached its quota. Displaying a local retrieval-based summary instead.*"
                answer = clarification_msg
        else:
            answer = self._generate_local_fallback_qa(user_query, retrieved)

        gen_latency = time.time() - gen_start
        print(f"[TIMING] Local Summary/QA generation took {gen_latency:.4f} seconds.")

        total_latency = time.time() - fallback_start
        print(f"[TIMING] Total local fallback execution took {total_latency:.4f} seconds.")

        trace = AgentTraceStep(
            step_index=1,
            thought="Gemini API unavailable/rate-limited. Executing local hybrid search and generating deterministic response.",
            tool_name="search_videos",
            tool_args={"query": user_query},
            observation=f"Observation: Found {len(retrieved)} relevant transcript segments."
        )
        self.trace_steps = [trace]

        return AgentExecutionResult(answer=answer, retrieved_chunks=retrieved, trace_steps=self.trace_steps, is_fallback=True)

    # ──────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────
    def run(self, user_query: str) -> AgentExecutionResult:
        """
        Runs the full ReAct agent orchestration loop for a given query.
        Returns a structured result containing the final answer, trace steps, and retrieved chunks.

        Always calls _get_client() to get a fresh runtime client — never uses a cached stale key.
        """
        query = user_query
        print("[AGENT INPUT]", query, flush=True)

        print(f"[AGENT.RUN] id(self.rag_manager): {id(self.rag_manager)}")
        print(f"[AGENT.RUN] len(self.rag_manager.chunks): {len(self.rag_manager.chunks)}")
        print(f"[AGENT.RUN] len(self.rag_manager.video_metadata_map): {len(self.rag_manager.video_metadata_map)}")

        print(len(self.rag_manager.chunks))
        print(len(self.rag_manager.video_metadata_map))
        print(list(self.rag_manager.video_metadata_map.keys()))

        original_query = user_query
        query_lower = user_query.lower()
        phrases = ["indexed video", "this video", "the video", "summarize the video", "summarize the indexed video"]
        if any(p in query_lower for p in phrases) and len(self.rag_manager.video_metadata_map) == 1:
            single_video_id = next(iter(self.rag_manager.video_metadata_map.keys()))
            query = f"Summarize the video with ID {single_video_id}."
            print("[QUERY BEFORE]", original_query, flush=True)
            print("[QUERY AFTER]", query, flush=True)
            logger.info(f"[QUERY REWRITE] Before: '{original_query}' -> After: '{query}'")
            user_query = query
        else:
            query = user_query

        print("[QUERY AFTER REWRITE]", query, flush=True)

        print(f"[{time.time():.3f}] [RUN] Step 1 entered")
        self.last_retrieved_chunks = []
        self.trace_steps = []

        run_start = time.time()
        logger.info(f"[TIMING] agent.run() START — query='{user_query}' at {run_start:.3f}")
        print(f"[TIMING] agent.run() START — query='{user_query}'")

        # 1. Resolve runtime client — reads os.environ at call-time
        client = self._get_client()

        if not client:
            logger.warning("[CHECKPOINT] FAIL — _get_client() returned None. GOOGLE_API_KEY missing from os.environ.")
            print("[CHECKPOINT] FAIL — _get_client() returned None. Running Demo/Mock fallback.")
            
            # Log Fallback entered timestamp
            print(f"[TIMING] Fallback entered at {time.time():.3f}", flush=True)
            logger.info(f"[TIMING] Fallback entered at {time.time():.3f}")
            
            result = self._run_mock_fallback(user_query)
            
            # Log Fallback completed timestamp
            print(f"[TIMING] Fallback completed at {time.time():.3f}", flush=True)
            logger.info(f"[TIMING] Fallback completed at {time.time():.3f}")
            
            elapsed = time.time() - run_start
            logger.info(f"[TIMING] agent.run() END (mock) — elapsed={elapsed:.2f}s")
            print(f"[TIMING] agent.run() END (mock) — elapsed={elapsed:.2f}s")
            print(f"[{time.time():.3f}] [RUN] Step 9 returning AgentResult")
            return result

        logger.info(f"[CHECKPOINT] PASS — _get_client() returned a valid client.")
        print(f"[CHECKPOINT] PASS — _get_client() returned a valid client.")

        # 2. Run real ReAct loop
        logger.info(f"[CHECKPOINT] Starting real Agent ReAct execution loop for query: '{user_query}'")

        def search_videos(query: str) -> str:
            """Search the hybrid RAG index for video transcript segments matching the query.

            Args:
                query: The search keywords or question.
            """
            return self._search_videos(query)

        def get_video_details(video_id: str) -> str:
            """Get metadata details for a specific video, including title, uploader, views, duration, and publish date.

            Args:
                video_id: The unique identifier of the video.
            """
            return self._get_video_details(video_id)

        def summarize_video(video_id: str) -> str:
            """Generate a high-level summary of the entire transcript for a specific video.

            Args:
                video_id: The unique identifier of the video to summarize.
            """
            return self._summarize_video(video_id)

        tools = [search_videos, get_video_details, summarize_video]

        messages = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_query)]
            )
        ]

        system_instruction = (
            "You are an Agentic Multi-Video Research Assistant.\n"
            "Your goal is to answer the user's research queries by finding and summarizing information from transcripts of videos (which may be in different languages).\n"
            "Always respond in the language of the user's question (default: English). Translate any retrieved context to match the user's query language if needed.\n"
            "You have access to the tools: 'search_videos', 'get_video_details', and 'summarize_video'.\n"
            "Always use 'search_videos' first to find relevant sections before trying to summarize or check details.\n"
            "When citing claims, you MUST use the exact format: [Video Title - MM:SS] where Video Title matches the exact title returned by the tool, and MM:SS matches the start timestamp.\n"
            "Reason step-by-step using thoughts before requesting any tool actions.\n"
            "Keep your final answer direct and factual. If the context does not contain the answer, say "
            "\"I do not know based on the provided video contexts.\"\n"
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.0
        )

        max_iterations = 10
        step_idx = 1

        while step_idx <= max_iterations:
            step_start = time.time()
            logger.info(f"[TIMING] ReAct Step {step_idx} START")
            print(f"[TIMING] ReAct Step {step_idx} START")

            try:
                logger.info(f"[CHECKPOINT] Calling Gemini generate_content — Step {step_idx}, model={GEMINI_LLM_MODEL}")
                print(f"[{time.time():.3f}] [RUN] Step 6 Gemini generate_content start")

                response = self.call_with_retry(
                    client.models.generate_content,
                    model=GEMINI_LLM_MODEL,
                    contents=messages,
                    config=config
                )

                print(f"[{time.time():.3f}] [RUN] Step 7 Gemini generate_content returned")
                step_elapsed = time.time() - step_start
                logger.info(f"[TIMING] ReAct Step {step_idx} Gemini call completed in {step_elapsed:.2f}s")
                print(f"[TIMING] ReAct Step {step_idx} Gemini call completed in {step_elapsed:.2f}s")

            except Exception as e:
                logger.warning(f"[CHECKPOINT] FAIL — Gemini API call failed: {e}. Falling back to local summary.")
                print(f"[CHECKPOINT] FAIL — Gemini API call failed. Falling back to local summary.")
                
                # Log Fallback entered timestamp
                print(f"[TIMING] Fallback entered at {time.time():.3f}", flush=True)
                logger.info(f"[TIMING] Fallback entered at {time.time():.3f}")
                
                result = self._run_mock_fallback(user_query)
                
                # Log Fallback completed timestamp
                print(f"[TIMING] Fallback completed at {time.time():.3f}", flush=True)
                logger.info(f"[TIMING] Fallback completed at {time.time():.3f}")
                
                elapsed = time.time() - run_start
                print(f"[TIMING] agent.run() END (fallback) — elapsed={elapsed:.2f}s")
                print(f"[{time.time():.3f}] [RUN] Step 9 returning AgentResult")
                return result

            # Extract thought
            thought = response.text or ""

            if response.function_calls:
                logger.info(f"[CHECKPOINT] Step {step_idx}: Model requested {len(response.function_calls)} tool call(s).")

                # Append the model's tool calls to messages history
                model_content = response.candidates[0].content
                if not model_content.role:
                    model_content.role = "model"
                messages.append(model_content)

                for call in response.function_calls:
                    tool_name = call.name
                    tool_args = call.args

                    logger.info(f"[CHECKPOINT] Executing tool: {tool_name}({tool_args})")
                    print(f"[CHECKPOINT] Executing tool: {tool_name}({tool_args})")

                    if tool_name == "search_videos":
                        observation = search_videos(**tool_args)
                    elif tool_name == "get_video_details":
                        observation = get_video_details(**tool_args)
                    elif tool_name == "summarize_video":
                        observation = summarize_video(**tool_args)
                    else:
                        observation = f"Observation: Error - Tool '{tool_name}' is not recognized."

                    self.trace_steps.append(AgentTraceStep(
                        step_index=step_idx,
                        thought=thought,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=observation
                    ))

                    response_part = types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation}
                    )
                    messages.append(types.Content(
                        role="tool",
                        parts=[response_part]
                    ))

                step_idx += 1

            else:
                logger.info(f"[CHECKPOINT] PASS — Step {step_idx}: Model returned final text response.")
                print(f"[CHECKPOINT] PASS — Step {step_idx}: Final answer produced.")

                self.trace_steps.append(AgentTraceStep(
                    step_index=step_idx,
                    thought=thought,
                    tool_name=None,
                    tool_args=None,
                    observation=None
                ))

                print(f"[{time.time():.3f}] [RUN] Step 8 final answer assembled")
                result = AgentExecutionResult(
                    answer=thought,
                    retrieved_chunks=self.last_retrieved_chunks,
                    trace_steps=self.trace_steps
                )

                elapsed = time.time() - run_start
                logger.info(f"[TIMING] agent.run() END (success) — elapsed={elapsed:.2f}s, answer_len={len(thought)}")
                print(f"[TIMING] agent.run() END (success) — elapsed={elapsed:.2f}s, answer_len={len(thought)}")
                print(f"[{time.time():.3f}] [RUN] Step 9 returning AgentResult")
                return result

        # Exceeded max iterations
        logger.warning("[CHECKPOINT] FAIL — ReAct loop exceeded maximum iteration limit.")
        elapsed = time.time() - run_start
        print(f"[TIMING] agent.run() END (timeout) — elapsed={elapsed:.2f}s")
        print(f"[{time.time():.3f}] [RUN] Step 9 returning AgentResult")
        return AgentExecutionResult(
            answer="Agent execution timed out without reaching a final conclusion.",
            retrieved_chunks=self.last_retrieved_chunks,
            trace_steps=self.trace_steps
        )


if __name__ == "__main__":
    agent = VideoResearchAgent()
    print("VideoResearchAgent core module initialized.")
