"""
test_agent.py — CLI verification runner for Phase 3.
Instantiates the VideoResearchAgent and runs multi-step reasoning queries,
printing out the detailed execution trace (thoughts, tools called, observations)
and the final cited answer.
"""

import sys
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from src.core.agent import VideoResearchAgent, AgentExecutionResult
from src.core.rag import format_time

def print_trace_result(result: AgentExecutionResult):
    """Utility to print the agent reasoning traces and final response."""
    print("\n" + "=" * 50)
    print(" AGENT EXECUTION TRACE LOGS")
    print("=" * 50)
    
    for step in result.trace_steps:
        print(f"\n[STEP {step.step_index}]")
        print(f"Thought: {step.thought.strip()}")
        if step.tool_name:
            print(f"Action: Call tool '{step.tool_name}' with args: {step.tool_args}")
            # Format observation preview
            obs_preview = step.observation.strip()
            if len(obs_preview) > 300:
                obs_preview = obs_preview[:300] + "\n  [... truncated for readability ...]"
            print(f"Observation:\n  {obs_preview}")
        else:
            print("Action: Final Answer reached.")
            
    print("\n" + "=" * 50)
    print(" FINAL ANSWER")
    print("=" * 50)
    print(result.answer)
    print("=" * 50)
    
    if result.retrieved_chunks:
        print("\nGrounding Context Chunks Used:")
        for idx, chunk in enumerate(result.retrieved_chunks, 1):
            print(f"  [{idx}] Video ID: {chunk.video_id} | Timestamp: {format_time(chunk.start_time)} - {format_time(chunk.end_time)}")

def main():
    print("=" * 60)
    print(" PHASE 3 TESTING: AGENTIC CORE & REPLAY TRACING ")
    print("=" * 60)
    
    agent = VideoResearchAgent()
    
    # Check if index is empty
    if not agent.rag_manager.chunks:
        print("[WARNING] Search index is empty. Please run Phase 1 test to index a video first.")
        print("          Running: python test_retrieval.py")
        sys.exit(1)
        
    # Query 1: Requires calling search_videos to find uploader details or searching and details lookup
    query_1 = "what is an API and who is the uploader of the overview video"
    print(f"\n>>> Running Query 1: '{query_1}'")
    try:
        res_1 = agent.run(query_1)
        print_trace_result(res_1)
    except Exception as e:
        print(f"[ERROR] Query 1 failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Query 2: Requires comparative reasoning or summarizing
    query_2 = "Summarize the video contents of ySEx_BqVx8A and compare the hybrid search explanation."
    print(f"\n>>> Running Query 2: '{query_2}'")
    try:
        res_2 = agent.run(query_2)
        print_trace_result(res_2)
    except Exception as e:
        print(f"[ERROR] Query 2 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(" PHASE 3 VERIFICATION RUN COMPLETED ")
    print("=" * 60)

if __name__ == "__main__":
    main()
