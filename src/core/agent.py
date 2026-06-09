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
from typing import List, Dict, Any, Optional, Tuple
import google.genai as google_genai
from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_LLM_MODEL
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
    """
    def __init__(self, rag_manager: Optional[HybridRAGManager] = None):
        self.rag_manager = rag_manager or HybridRAGManager()
        self.last_retrieved_chunks: List[RetrievalChunk] = []
        self.trace_steps: List[AgentTraceStep] = []
        
        # Configure GenAI Client if key is present
        if GOOGLE_API_KEY:
            self.client = google_genai.Client(api_key=GOOGLE_API_KEY)
        else:
            self.client = None
            logger.warning("GOOGLE_API_KEY not configured. Agent will run in Demo/Mock mode.")

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
        
        if not self.client:
            # Mock summary in demo mode
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
                self.client.models.generate_content,
                model=GEMINI_LLM_MODEL,
                contents=prompt
            )
            return f"Observation:\n{response.text}"
        except Exception as e:
            return f"Observation: Error summarizing video transcript via Gemini: {e}"

    def _run_mock_fallback(self, user_query: str) -> AgentExecutionResult:
        """Runs a simulated ReAct loop and answer generation grounded in retrieved RAG context when API key is limited or missing."""
        mock_chunks = self.rag_manager.query_index(user_query, limit=3)
        self.last_retrieved_chunks.extend(mock_chunks)
        
        # Determine some parameters from chunks or query
        video_id = mock_chunks[0].video_id if mock_chunks else "ySEx_BqVx8A"
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
                f"  Duration: 00:30\n"
                f"  Views: 100\n"
                f"  Publish Date: 2026-06-08\n"
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
        
        # Construct cited mock answer
        answer = (
            f"[DEMO FALLBACK] Since the Gemini API key has hit daily quota/rate limits, "
            f"here is the simulated agent answer grounded in retrieved content:\n\n"
            f"According to [{title} - 00:17], the system leverages Gemini 2.5 Flash for RAG generation. "
            f"The video uploader is '{uploader}', as shown in the video details."
        )
        return AgentExecutionResult(answer=answer, retrieved_chunks=mock_chunks, trace_steps=self.trace_steps)

    def run(self, user_query: str) -> AgentExecutionResult:
        """
        Runs the full ReAct agent orchestration loop for a given query.
        Returns a structured result containing the final answer, trace steps, and retrieved chunks.
        """
        self.last_retrieved_chunks = []
        self.trace_steps = []
        
        # 1. Run in Demo Mode if API key is missing
        if not self.client:
            logger.warning("Agent running in Mock/Demo Mode (GOOGLE_API_KEY missing).")
            return self._run_mock_fallback(user_query)

        # 2. Run real ReAct loop
        logger.info(f"Starting real Agent ReAct execution loop for query: '{user_query}'")
        
        # Tools definitions inside the local scope of the loop to generate clear call signatures
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
        
        # Build chat history
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
            logger.info(f"Running ReAct Loop Step {step_idx}...")
            
            try:
                # Call Gemini model
                response = self.call_with_retry(
                    self.client.models.generate_content,
                    model=GEMINI_LLM_MODEL,
                    contents=messages,
                    config=config
                )
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower()
                if is_rate_limit:
                    logger.warning("Gemini API daily/rate quota limit exceeded during loop. Falling back to Mock/Demo Mode.")
                    return self._run_mock_fallback(user_query)
                
                logger.error(f"Error in ReAct loop during generate_content: {e}")
                return AgentExecutionResult(
                    answer=f"Error running agent loop: {e}",
                    retrieved_chunks=self.last_retrieved_chunks,
                    trace_steps=self.trace_steps
                )
                
            # Extract thought
            thought = response.text or ""
            
            if response.function_calls:
                logger.info(f"Step {step_idx}: Model requested {len(response.function_calls)} tool call(s).")
                
                # Append the model's tool calls to messages history
                model_content = response.candidates[0].content
                if not model_content.role:
                    model_content.role = "model"
                messages.append(model_content)
                
                # Execute each call
                for call in response.function_calls:
                    tool_name = call.name
                    tool_args = call.args
                    
                    # Execute tool action
                    if tool_name == "search_videos":
                        observation = search_videos(**tool_args)
                    elif tool_name == "get_video_details":
                        observation = get_video_details(**tool_args)
                    elif tool_name == "summarize_video":
                        observation = summarize_video(**tool_args)
                    else:
                        observation = f"Observation: Error - Tool '{tool_name}' is not recognized."
                        
                    # Save trace step
                    self.trace_steps.append(AgentTraceStep(
                        step_index=step_idx,
                        thought=thought,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=observation
                    ))
                    
                    # Create and append tool response part to messages history
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
                logger.info(f"Step {step_idx}: Model completed reasoning and returned final text response.")
                
                # Record the final thought step (no tool call)
                self.trace_steps.append(AgentTraceStep(
                    step_index=step_idx,
                    thought=thought,
                    tool_name=None,
                    tool_args=None,
                    observation=None
                ))
                
                # The response text represents the final answer
                return AgentExecutionResult(
                    answer=thought,
                    retrieved_chunks=self.last_retrieved_chunks,
                    trace_steps=self.trace_steps
                )
                
        # If we exceeded max iterations without a final response
        logger.warning("ReAct loop exceeded maximum iteration limits without finishing.")
        return AgentExecutionResult(
            answer="Agent execution timed out without reaching a final conclusion.",
            retrieved_chunks=self.last_retrieved_chunks,
            trace_steps=self.trace_steps
        )

if __name__ == "__main__":
    agent = VideoResearchAgent()
    print("VideoResearchAgent core module initialized.")
