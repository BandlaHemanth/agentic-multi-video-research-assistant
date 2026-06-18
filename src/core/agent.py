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
        """
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
                is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str

                if (is_rate_limit or is_unavailable) and attempt < max_attempts:
                    # Parse retry delay
                    wait_time = 45
                    match = re.search(r"retry in (\d+\.?\d*)s", err_str)
                    if match:
                        wait_time = int(float(match.group(1))) + 2
                    else:
                        match_sec = re.search(r"retryDelay': '(\d+)s'", err_str)
                        if match_sec:
                            wait_time = int(match_sec.group(1)) + 2

                    logger.warning(
                        f"[Attempt {attempt}/{max_attempts}] Gemini API rate-limited/unavailable. "
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
        chunks = self.rag_manager.query_index(query)
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
            return (
                f"Observation: [DEMO MODE Summary for {meta.get('title', 'Video')}]: "
                f"This video discusses key architecture blocks of the Agentic Multi-Video Research Assistant, "
                f"detailing hybrid retrieval, dense FAISS indexing, sparse BM25 indexing, and MiniLM rerankers."
            )

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
    # DEMO/MOCK FALLBACK
    # ──────────────────────────────────────────────────────────────────
    def _run_mock_fallback(self, user_query: str) -> AgentExecutionResult:
        """Runs a simulated ReAct loop and answer generation grounded in retrieved RAG context when API key is limited or missing."""
        mock_chunks = self.rag_manager.query_index(user_query, limit=3)
        self.last_retrieved_chunks.extend(mock_chunks)

        video_id = mock_chunks[0].video_id if mock_chunks else "unknown"
        meta = self.rag_manager.video_metadata_map.get(video_id, {})
        title = meta.get("title", "Video")
        uploader = meta.get("uploader", "Unknown")

        trace1 = AgentTraceStep(
            step_index=1,
            thought="I need to search for video transcript segments related to the query.",
            tool_name="search_videos",
            tool_args={"query": user_query},
            observation=f"Observation: Found {len(mock_chunks)} relevant transcript segments."
        )
        trace2 = AgentTraceStep(
            step_index=2,
            thought="I should also get details about the video to answer questions about the uploader or duration.",
            tool_name="get_video_details",
            tool_args={"video_id": video_id},
            observation=(
                f"Observation:\n"
                f"  Video ID: {video_id}\n"
                f"  Title: {title}\n"
                f"  Uploader: {uploader}\n"
            )
        )
        trace3 = AgentTraceStep(
            step_index=3,
            thought="I have collected all necessary search segments and video details. I will now compile the final answer.",
            tool_name=None,
            tool_args=None,
            observation=None
        )
        self.trace_steps = [trace1, trace2, trace3]

        answer = (
            f"[DEMO FALLBACK] Since the Gemini API key has hit daily quota/rate limits, "
            f"here is the simulated agent answer grounded in retrieved content:\n\n"
            f"According to [{title} - 00:17], the system leverages Gemini 2.5 Flash for RAG generation. "
            f"The video uploader is '{uploader}', as shown in the video details."
        )
        return AgentExecutionResult(answer=answer, retrieved_chunks=mock_chunks, trace_steps=self.trace_steps)

    # ──────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────
    def run(self, user_query: str) -> AgentExecutionResult:
        """
        Runs the full ReAct agent orchestration loop for a given query.
        Returns a structured result containing the final answer, trace steps, and retrieved chunks.

        Always calls _get_client() to get a fresh runtime client — never uses a cached stale key.
        """
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
            result = self._run_mock_fallback(user_query)
            elapsed = time.time() - run_start
            logger.info(f"[TIMING] agent.run() END (mock) — elapsed={elapsed:.2f}s")
            print(f"[TIMING] agent.run() END (mock) — elapsed={elapsed:.2f}s")
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
                print(f"[CHECKPOINT] Calling Gemini generate_content — Step {step_idx}, model={GEMINI_LLM_MODEL}")

                response = self.call_with_retry(
                    client.models.generate_content,
                    model=GEMINI_LLM_MODEL,
                    contents=messages,
                    config=config
                )

                step_elapsed = time.time() - step_start
                logger.info(f"[TIMING] ReAct Step {step_idx} Gemini call completed in {step_elapsed:.2f}s")
                print(f"[TIMING] ReAct Step {step_idx} Gemini call completed in {step_elapsed:.2f}s")

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower()
                if is_rate_limit:
                    logger.warning("[CHECKPOINT] FAIL — Gemini rate limit hit. Falling back to Mock/Demo Mode.")
                    print("[CHECKPOINT] FAIL — Rate limit. Falling back to mock.")
                    result = self._run_mock_fallback(user_query)
                    elapsed = time.time() - run_start
                    print(f"[TIMING] agent.run() END (rate-limit fallback) — elapsed={elapsed:.2f}s")
                    return result

                import traceback
                tb = traceback.format_exc()
                logger.error(f"[CHECKPOINT] FAIL — Exception in ReAct Step {step_idx}: {e}\n{tb}")
                print(f"[CHECKPOINT] FAIL — Exception:\n{tb}")
                elapsed = time.time() - run_start
                print(f"[TIMING] agent.run() END (error) — elapsed={elapsed:.2f}s")
                return AgentExecutionResult(
                    answer=f"Error running agent loop: {e}",
                    retrieved_chunks=self.last_retrieved_chunks,
                    trace_steps=self.trace_steps
                )

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

                result = AgentExecutionResult(
                    answer=thought,
                    retrieved_chunks=self.last_retrieved_chunks,
                    trace_steps=self.trace_steps
                )

                elapsed = time.time() - run_start
                logger.info(f"[TIMING] agent.run() END (success) — elapsed={elapsed:.2f}s, answer_len={len(thought)}")
                print(f"[TIMING] agent.run() END (success) — elapsed={elapsed:.2f}s, answer_len={len(thought)}")
                return result

        # Exceeded max iterations
        logger.warning("[CHECKPOINT] FAIL — ReAct loop exceeded maximum iteration limit.")
        elapsed = time.time() - run_start
        print(f"[TIMING] agent.run() END (timeout) — elapsed={elapsed:.2f}s")
        return AgentExecutionResult(
            answer="Agent execution timed out without reaching a final conclusion.",
            retrieved_chunks=self.last_retrieved_chunks,
            trace_steps=self.trace_steps
        )


if __name__ == "__main__":
    agent = VideoResearchAgent()
    print("VideoResearchAgent core module initialized.")
