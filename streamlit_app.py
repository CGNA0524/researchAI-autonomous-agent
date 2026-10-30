"""
Streamlit Demo Interface for the Predixion AI Research Agent.
Provides an interactive interface showing the agent's plan, tool executions,
sources, and final structured report.
"""

import asyncio
import json
import streamlit as st
from app.agent.orchestrator import AgentOrchestrator
from app.config import get_settings

st.set_page_config(
    page_title="AI Research Agent | Predixion AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refresh settings on every page rerun to dynamically catch .env edits
get_settings.cache_clear()
settings = get_settings()

st.title("🔍 Autonomous AI Research Agent")
st.caption("Predixion AI Platform Engineering Internship Demonstration")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    active_prov = settings.LLM_PROVIDER.lower().strip()
    st.markdown(f"**Provider:** `{active_prov.upper()}`")
    if active_prov == "openai":
        st.markdown(f"**Model:** `{settings.OPENAI_MODEL}`")
        st.markdown(f"**Max Budget Cap:** `${settings.MAX_REQUEST_BUDGET_USD:.2f}`")
    else:
        st.markdown(f"**Primary:** `{settings.GROQ_MODEL}`")
        st.markdown(f"**Fallback:** `{settings.GROQ_FALLBACK_MODEL}`")
    st.markdown(f"**Timeout:** `{settings.HTTP_TIMEOUT_SECONDS}s`")
    
    st.divider()
    st.subheader("Preset Demo Queries")
    demo_queries = [
        "Compare the top 3 open-source vector databases for a startup building RAG products.",
        "What are the main security vulnerabilities in MCP (Model Context Protocol) servers?",
        "Compare DuckDB vs SQLite for local in-process analytical workloads in Python.",
        "Unicorns teleportation quantum cryptography in 1492 (Edge Case: Insufficient Info)",
    ]
    selected_demo = st.selectbox("Select a sample query:", ["Custom Query"] + demo_queries)

# Main Query Input
default_input = "" if selected_demo == "Custom Query" else selected_demo
query_text = st.text_area(
    "Enter Research Question:",
    value=default_input,
    placeholder="e.g. Compare the top 3 open-source vector databases for a startup building RAG products.",
    height=100
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("🚀 Conduct Research", type="primary", use_container_width=True)

if run_button:
    if not query_text.strip():
        st.warning("Please enter a research question first.")
    else:
        st.info(f"Researching: **{query_text.strip()}**")
        
        with st.status("Agent Workflow in Progress...", expanded=True) as status:
            st.write("📋 **Step 1:** Understanding question and formulating research plan...")
            st.write("🛠️ **Step 2:** Selecting tools (Web Search & Webpage Extraction)...")
            st.write("🌐 **Step 3:** Executing web searches and retrieving live evidence...")
            st.write("🛡️ **Step 4:** Applying SSRF validation, prompt injection defense, and deduplication...")
            st.write("⚖️ **Step 5:** Assessing confidence, limitations, and synthesizing grounded answer...")
            
            orchestrator = AgentOrchestrator()
            
            # Run async orchestrator inside streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(orchestrator.run(query_text.strip()))
                status.update(label="✅ Research Workflow Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Error during research execution", state="error", expanded=True)
                st.error(f"Execution Error: {str(e)}")
                response = None
            finally:
                loop.close()

        if response:
            st.success("### Executive Short Answer")
            st.markdown(response.short_answer)

            # Metadata Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Confidence Level", response.confidence_level)
            with c2:
                st.metric("Verified Sources", len(response.sources_used))
            with c3:
                time_taken = response.execution_trace.total_execution_time_seconds if response.execution_trace else 0.0
                st.metric("Execution Time", f"{time_taken}s")
            with c4:
                trace = response.execution_trace
                if trace and trace.token_usage:
                    cost_val = trace.token_usage.estimated_cost_usd
                    cost_str = f"${cost_val:.5f}" if cost_val is not None else "Free / Unpriced"
                    st.metric(
                        "Tokens Used",
                        f"{trace.token_usage.total_tokens:,} tok",
                        f"Est: {cost_str}"
                    )
                else:
                    st.metric("Provider Used", (trace.llm_provider_used if trace else active_prov).upper())

            # Confidence Explanation
            if response.confidence_explanation:
                st.info(f"**Confidence Analysis:** {response.confidence_explanation}")

            # Evidence Coverage Matrix (if available)
            if response.evidence_coverage and response.evidence_coverage.get("coverage_matrix"):
                with st.expander("📊 Evidence Coverage Matrix across Entities & Dimensions", expanded=True):
                    cov = response.evidence_coverage
                    st.write(f"**Coverage Ratio:** {int(cov.get('coverage_ratio', 0) * 100)}% | **Covered:** {cov.get('covered_cells')}/{cov.get('total_cells')} dimensions")
                    matrix_data = cov.get("coverage_matrix", {})
                    if matrix_data:
                        st.table(matrix_data)

            # Key Findings
            st.subheader("💡 Key Findings")
            for idx, finding in enumerate(response.key_findings, 1):
                st.markdown(f"**{idx}.** {finding}")

            # Sources Used & Authority Classification
            st.subheader("📚 Verified Sources Used")
            if response.sources_used:
                for src in response.sources_used:
                    badge = f" [{src.category}]" if src.category else ""
                    official_badge = " ⭐ (Official)" if src.is_official else ""
                    with st.expander(f"🔗 {src.title}{badge}{official_badge}"):
                        st.markdown(f"**URL:** [{src.url}]({src.url})")
                        st.markdown(f"**Relevance:** {src.relevance}")
            else:
                st.write("No external sources could be retrieved.")

            # Source Classification Breakdown
            if response.source_classification:
                with st.expander("🏛️ Source Authority Breakdown"):
                    for category, items in response.source_classification.items():
                        st.markdown(f"**{category}** ({len(items)})")
                        for item in items:
                            st.markdown(f"- {item}")

            # Assumptions & Limitations
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("📌 Key Assumptions")
                for a in response.assumptions:
                    st.markdown(f"- {a}")
            with col_b:
                st.subheader("⚠️ Limitations")
                for l in response.limitations:
                    st.markdown(f"- {l}")

            # Suggested Next Steps
            st.subheader("🎯 Suggested Next Steps")
            for step in response.suggested_next_steps:
                st.markdown(f"- {step}")

            # Observability & Execution Trace
            if response.execution_trace:
                with st.expander("🔬 Agent Orchestration Trace (Observability & Retrieval Rounds)"):
                    trace_dump = response.execution_trace.model_dump()
                    st.write(f"**Retrieval Rounds:** {trace_dump.get('retrieval_rounds', 1)}")
                    if trace_dump.get("round_2_queries"):
                        st.write(f"**Second-Hop Queries:** {trace_dump.get('round_2_queries')}")
                    st.json(trace_dump)

            # Raw Structured JSON output
            with st.expander("📄 Raw Validated JSON Output (Pydantic Schema)"):
                st.code(json.dumps(response.model_dump(exclude={"execution_trace"}), indent=2), language="json")
