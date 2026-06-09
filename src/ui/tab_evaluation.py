"""
tab_evaluation.py — Evaluation Dashboard UI tab (SaaS Redesign).
Displays historical evaluation records and metrics (Faithfulness, Relevancy, etc.),
allows running a new RAGAS evaluation on the last generated chat assistant response,
and renders visual trend charts.
"""

import pandas as pd
import streamlit as st
import logging
from typing import List, Dict, Any

from src.core.agent import VideoResearchAgent
from src.core.rag import HybridRAGManager
from src.evaluation.evaluator import evaluate_rag_query, get_evaluation_history

logger = logging.getLogger(__name__)

def render_evaluation_tab(agent: VideoResearchAgent, rag_manager: HybridRAGManager):
    """
    Renders the RAGAS Evaluation Dashboard.
    """
    st.subheader("📊 Evaluation Dashboard")
    st.markdown("*Inspect retrieval and generation metrics evaluated using RAGAS (Faithfulness, Answer Relevancy, and Context Precision).*")
    
    # ────────────────────────────────────────────────────────────────
    # 1. EVALUATE LAST QUERY ACTION
    # ────────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Evaluate Last Response")
    
    # Check if there is chat history to evaluate
    history = st.session_state.get("chat_history", [])
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    user_msgs = [m for m in history if m["role"] == "user"]
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if not assistant_msgs or not user_msgs:
        st.info("No queries have been run in this session. Ask a question in the Chat Assistant tab to run an evaluation.")
    else:
        last_user_query = user_msgs[-1]["content"]
        last_assistant_msg = assistant_msgs[-1]
        last_ans = last_assistant_msg["content"]
        
        # Get contexts
        retrieved_chunks = last_assistant_msg.get("retrieved_chunks", [])
        contexts = [c.get("text", "") for c in retrieved_chunks]
        
        st.write(f"**Last Query:** *\"{last_user_query}\"*")
        
        # Evaluate Button
        eval_btn = st.button("Evaluate Response Quality", key="eval_run_btn", help="Run RAGAS evaluation on the last exchange")
        
        if eval_btn:
            if not contexts:
                st.warning("No grounding contexts were retrieved for this query. Faithfulness and Precision cannot be calculated.")
                contexts = ["No context available."]
                
            with st.spinner("Computing RAGAS metrics using Gemini..."):
                try:
                    scores = evaluate_rag_query(
                        query=last_user_query,
                        answer=last_ans,
                        contexts=contexts
                    )
                    
                    st.success("Evaluation complete!")
                    st.toast("RAGAS Evaluation complete.")
                    
                    # Render scores
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        f_score = scores.get("faithfulness")
                        st.metric("Faithfulness", f"{f_score:.2f}" if f_score is not None else "N/A")
                    with col_s2:
                        r_score = scores.get("answer_relevancy")
                        st.metric("Answer Relevancy", f"{r_score:.2f}" if r_score is not None else "N/A")
                    with col_s3:
                        p_score = scores.get("context_precision")
                        st.metric("Context Precision", f"{p_score:.2f}" if p_score is not None else "N/A")
                        
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to calculate RAGAS metrics: {e}")
                    logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────
    # 2. HISTORICAL METRICS SUMMARY & CHARTS
    # ────────────────────────────────────────────────────────────────
    st.write("---")
    st.markdown("### 📈 Historical Metric Trends")
    
    eval_history = get_evaluation_history()
    
    if not eval_history:
        st.info("No prior evaluation history found. Complete an evaluation to view dashboard metrics.")
        return
        
    records = []
    for item in eval_history:
        scores = item.get("scores", {})
        records.append({
            "Timestamp": pd.to_datetime(item.get("timestamp")),
            "Query": item.get("query"),
            "Faithfulness": scores.get("faithfulness", 0.0),
            "Answer Relevancy": scores.get("answer_relevancy", 0.0),
            "Context Precision": scores.get("context_precision", 0.0),
        })
        
    df = pd.DataFrame(records)
    
    # Calculate Average Metrics
    avg_f = df["Faithfulness"].mean()
    avg_r = df["Answer Relevancy"].mean()
    avg_p = df["Context Precision"].mean()
    
    # Render Average Metric Cards
    col_av1, col_av2, col_av3 = st.columns(3)
    with col_av1:
        st.markdown(
            f'<div class="glass-card" style="text-align:center; border-left: 5px solid #22c55e;">'
            f'<div class="metric-label">Avg Faithfulness</div>'
            f'<div class="metric-value" style="color:#22c55e;">{avg_f:.2f}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )
    with col_av2:
        st.markdown(
            f'<div class="glass-card" style="text-align:center; border-left: 5px solid #7c5cfc;">'
            f'<div class="metric-label">Avg Relevancy</div>'
            f'<div class="metric-value" style="color:#7c5cfc;">{avg_r:.2f}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )
    with col_av3:
        st.markdown(
            f'<div class="glass-card" style="text-align:center; border-left: 5px solid #0ea5e9;">'
            f'<div class="metric-label">Avg Precision</div>'
            f'<div class="metric-value" style="color:#0ea5e9;">{avg_p:.2f}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )
        
    # Render Trend Line Chart
    st.markdown("#### Score Progression")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    chart_df = df.set_index("Timestamp")[["Faithfulness", "Answer Relevancy", "Context Precision"]]
    st.line_chart(chart_df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────
    # 3. DETAILED LOGS GRID
    # ────────────────────────────────────────────────────────────────
    st.markdown("#### Detailed Logs")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    display_df = df.copy()
    display_df["Timestamp"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
