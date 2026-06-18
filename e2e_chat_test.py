"""
e2e_chat_test.py — End-to-end verification of the full chat pipeline.

Stages tested:
  1. os.environ key resolution
  2. VideoResearchAgent._get_client()
  3. Video ingestion (Rick Astley dQw4w9WgXcQ)
  4. RAG indexing
  5. agent.run("Summarize the indexed video.")
  6. Final answer rendered

Run with:
  .venv\\Scripts\\python.exe e2e_chat_test.py AIzaSy...your_key_here...
"""

import sys
import os
import time

if len(sys.argv) < 2:
    print("Usage: python e2e_chat_test.py <YOUR_GEMINI_API_KEY>")
    sys.exit(1)

api_key = sys.argv[1].strip()
if len(api_key) < 20:
    print(f"ERROR: Key is too short. Got: {len(api_key)} chars.")
    print("This is not a valid Gemini API key.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = api_key
print(f"[CHECKPOINT] PASS — A: os.environ['GOOGLE_API_KEY'] set ({api_key[:12]}...)")

# ─── Step 1: Import modules ───────────────────────────────────────────────────
from src.core.agent import VideoResearchAgent
from src.core.rag import HybridRAGManager
from src.core.ingestion import ingest_video

# ─── Step 2: Resolve client ──────────────────────────────────────────────────
rag = HybridRAGManager()
agent = VideoResearchAgent(rag_manager=rag)

client = agent._get_client()
if client is None:
    print("[CHECKPOINT] FAIL — B: _get_client() returned None. Key not in os.environ.")
    sys.exit(1)
print("[CHECKPOINT] PASS — B: _get_client() returned a valid google.genai.Client")

# ─── Step 3: Ingest Rick Astley video ────────────────────────────────────────
RICK_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
print(f"\n[CHECKPOINT] C: Ingesting video: {RICK_URL}")
t0 = time.time()
try:
    video_data = ingest_video(RICK_URL, progress_callback=lambda m: print(f"  [ingest] {m}"))
    elapsed = time.time() - t0
    print(f"[CHECKPOINT] PASS — C: Ingested '{video_data.metadata.title}' in {elapsed:.1f}s")
except Exception as e:
    import traceback
    print(f"[CHECKPOINT] FAIL — C: Ingestion raised exception:\n{traceback.format_exc()}")
    sys.exit(1)

# ─── Step 4: Index into RAG ───────────────────────────────────────────────────
print("\n[CHECKPOINT] D: Adding video to RAG index...")
t0 = time.time()
try:
    rag.add_video(video_data)
    elapsed = time.time() - t0
    print(f"[CHECKPOINT] PASS — D: {len(rag.chunks)} chunks indexed in {elapsed:.1f}s")
except Exception as e:
    import traceback
    print(f"[CHECKPOINT] FAIL — D: rag.add_video() raised:\n{traceback.format_exc()}")
    sys.exit(1)

# ─── Step 5: Run agent query ──────────────────────────────────────────────────
QUERY = "Summarize the indexed video."
print(f"\n[CHECKPOINT] E: Calling agent.run('{QUERY}')")
t0 = time.time()
try:
    result = agent.run(QUERY)
    elapsed = time.time() - t0
    print(f"[CHECKPOINT] PASS — E: agent.run() returned in {elapsed:.1f}s")
except Exception as e:
    import traceback
    print(f"[CHECKPOINT] FAIL — E: agent.run() raised:\n{traceback.format_exc()}")
    sys.exit(1)

# ─── Step 6: Verify answer ────────────────────────────────────────────────────
answer = result.answer
if not answer or len(answer) < 20:
    print(f"[CHECKPOINT] FAIL — F: Answer is empty or too short: '{answer}'")
    sys.exit(1)

print(f"[CHECKPOINT] PASS — F: answer_len={len(answer)}")

# Check it is NOT the demo fallback
if "[DEMO FALLBACK]" in answer:
    print("[CHECKPOINT] FAIL — G: Answer is a DEMO FALLBACK — real Gemini never called.")
    print("  This means _get_client() returned None during agent.run() even though the key was set.")
    print("  Check that os.environ is still set at the time agent.run() executes.")
    sys.exit(1)

print("[CHECKPOINT] PASS — G: Answer is a REAL Gemini response (not demo fallback)")

print("\n" + "="*60)
print(" END-TO-END TEST PASSED")
print("="*60)
print(f"\nFINAL ANSWER ({len(answer)} chars):\n")
print(answer[:1000])
if len(answer) > 1000:
    print(f"\n... [{len(answer)-1000} more chars]")

print(f"\nTrace steps: {len(result.trace_steps)}")
for step in result.trace_steps:
    tool_info = f" → tool={step.tool_name}" if step.tool_name else " → [final answer]"
    print(f"  Step {step.step_index}{tool_info}")
