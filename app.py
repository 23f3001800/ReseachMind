# app.py
import streamlit as st
from dotenv import load_dotenv
from agent.graph import build_graph
from agent.tools import build_rag_retriever
from langchain_community.document_loaders import PyPDFLoader
import tempfile, os

load_dotenv()

st.set_page_config(page_title="ResearchMind — AI Agent", page_icon="🔬", layout="wide")
st.title("🔬 ResearchMind")
st.caption("Give it a topic. Get a research report in 60 seconds.")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    depth = st.selectbox("Research depth", ["quick", "deep"])

    st.subheader("Knowledge base (optional)")
    uploaded = st.file_uploader("Upload a PDF to include in research", type="pdf")
    retriever = None
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read())
            tmp = f.name
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(tmp).load()
        retriever = build_rag_retriever(docs)
        st.success(f"✓ Loaded {len(docs)} pages into knowledge base")

# ── Main area ────────────────────────────────────────────────
topic = st.text_input(
    "Research topic",
    placeholder="e.g. Impact of LLMs on software engineering jobs in 2026"
)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("Research →", type="primary", disabled=not topic)

if run and topic:
    graph = build_graph(retriever=retriever)

    # Progress tracking
    progress = st.progress(0)
    status   = st.empty()
    steps_el = st.empty()

    with st.spinner("Agent is researching..."):
        # Stream graph execution step by step
        final_state = None
        step_count  = 0
        node_labels  = {
            "planner":    ("Planning search strategy...",    20),
            "search":     ("Searching the web...",           40),
            "extract":    ("Extracting key facts...",        60),
            "rag":        ("Querying knowledge base...",     75),
            "synthesise": ("Writing report...",              90),
        }

        for event in graph.stream(
            {"topic": topic, "depth": depth, "steps_taken": []},
            stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                if node_name in node_labels:
                    label, pct = node_labels[node_name]
                    status.markdown(f"**{label}**")
                    progress.progress(pct)
                    if node_output.get("steps_taken"):
                        steps_el.caption(" → ".join(node_output["steps_taken"][-3:]))
                final_state = node_output

        progress.progress(100)
        status.empty()
        steps_el.empty()

    # ── Display results ───────────────────────────────────────
    if final_state and final_state.get("final_report"):
        st.markdown("---")

        # Report
        st.markdown(final_state["final_report"])

        # Citations
        if final_state.get("citations"):
            with st.expander(f"📚 {len(final_state['citations'])} sources"):
                for i, url in enumerate(final_state["citations"][:8], 1):
                    st.markdown(f"{i}. [{url}]({url})")

        # Download
        st.download_button(
            "⬇ Download report",
            data=final_state["final_report"],
            file_name=f"research_{topic[:30].replace(' ', '_')}.md",
            mime="text/markdown"
        )
    else:
        st.error("Research failed. Check your API keys and try again.")