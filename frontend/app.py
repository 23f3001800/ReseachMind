import streamlit as st
import requests
import json
import time
import sseclient

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS — Premium Dark Theme ──────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    /* Status container */
    [data-testid="stStatusWidget"] {
        border-radius: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.2) !important;
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h1 {
        font-size: 1.3rem !important;
        background: none;
        -webkit-text-fill-color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
    thread_id = st.text_input("Thread ID (session)", value="default")
    use_streaming = st.toggle("🔴 Live Streaming", value=True)

    st.divider()
    if st.button("🔍 Health Check"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            st.success("API online ✅") if r.ok else st.error(r.text)
        except Exception as e:
            st.error(f"Unreachable: {e}")

    st.divider()
    if st.button("🧠 View Graph Structure"):
        try:
            r = requests.get(f"{api_url}/agent/graph", timeout=5)
            if r.ok:
                st.json(r.json())
        except Exception as e:
            st.error(str(e))

    st.divider()
    if st.button("🗑️ Clear Memory"):
        try:
            r = requests.delete(f"{api_url}/agent/history/{thread_id}", timeout=5)
            if r.ok:
                st.success("Memory cleared.")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.caption("Agent Flow")
    st.markdown("""
    ```
    User Query
        ↓
    Researcher Agent  ←─┐
        ↓               │
    Analyst Agent ──────┘ (retry if gaps)
        ↓
    Writer Agent
        ↓
    Structured Report
    ```
    """)

# ── Main ──────────────────────────────────────────────────
st.title("🤖 Agentic Research Assistant")
st.caption("Multi-agent system: Researcher → Analyst → Writer · Memory per session · Guardrails enabled · Live streaming")

tabs = st.tabs(["💬 Research", "📁 Documents", "📜 History", "📊 About"])

# ── Tab 1: Research ───────────────────────────────────────
with tabs[0]:
    query = st.text_area(
        "Enter your research query",
        height=100,
        placeholder="e.g. What are the latest advances in multi-agent AI systems?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run = st.button("🚀 Run Agents", use_container_width=True)

    if run:
        if not query.strip():
            st.warning("Please enter a research query.")
        elif use_streaming:
            # ── Streaming mode ────────────────────────────
            with st.status("Running multi-agent pipeline...", expanded=True) as status:
                agent_status = st.empty()
                start_time = time.time()

                try:
                    response = requests.post(
                        f"{api_url}/agent/chat/stream",
                        json={"message": query, "thread_id": thread_id},
                        stream=True,
                        timeout=180,
                    )

                    if not response.ok:
                        status.update(label="❌ Failed", state="error")
                        st.error(f"Error: {response.text}")
                    else:
                        final_data = None
                        agent_icons = {
                            "researcher": "🔍",
                            "analyst": "📊",
                            "writer": "✍️",
                        }

                        for line in response.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data: "):
                                continue

                            try:
                                event_data = json.loads(line[6:])  # strip "data: "
                            except json.JSONDecodeError:
                                continue

                            event_type = event_data.get("event")
                            agent_name = event_data.get("agent", "")
                            icon = agent_icons.get(agent_name, "⚡")

                            if event_type == "agent_end":
                                conf = event_data.get("data", {}).get("confidence")
                                conf_str = f" (confidence: {conf:.2f})" if conf else ""
                                st.write(f"{icon} **{agent_name.title()}** completed{conf_str}")

                            elif event_type == "complete":
                                final_data = event_data.get("data", {})

                            elif event_type == "error":
                                status.update(label="❌ Error", state="error")
                                st.error(f"Pipeline error: {event_data.get('content')}")

                        if final_data and final_data.get("report"):
                            status.update(label="✅ Report ready!", state="complete")
                            report = final_data["report"]

                            # Guardrail warning
                            if final_data.get("needs_human_review"):
                                st.warning(
                                    f"⚠️ Human review recommended — "
                                    f"confidence: {final_data.get('confidence', 0):.2f}"
                                )

                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Confidence", f"{final_data.get('confidence', 0):.2f}")
                            col2.metric("Latency", f"{final_data.get('latency_ms', 0):.0f} ms")
                            col3.metric("Iterations", final_data.get("iterations", 0))

                            st.divider()

                            # Report sections
                            st.subheader(report.get("title", "Report"))
                            st.markdown(f"**Summary:** {report.get('summary', '')}")

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.markdown("**🔍 Key Findings**")
                                for f in report.get("research_findings", []):
                                    if f:
                                        st.markdown(f"- {f}")

                            with col_b:
                                st.markdown("**📊 Analysis**")
                                for a in report.get("analysis", []):
                                    if a:
                                        st.markdown(f"- {a}")

                            st.markdown("**Conclusion**")
                            st.write(report.get("conclusion", ""))

                            sources = final_data.get("sources", [])
                            if sources:
                                with st.expander("📎 Sources"):
                                    for s in sources:
                                        st.write(f"- {s}")

                            # Copy report button
                            full_report_text = f"""# {report.get('title', 'Report')}

## Summary
{report.get('summary', '')}

## Key Findings
""" + "\n".join(f"- {f}" for f in report.get("research_findings", [])) + f"""

## Analysis
""" + "\n".join(f"- {a}" for a in report.get("analysis", [])) + f"""

## Conclusion
{report.get('conclusion', '')}
"""
                            st.download_button(
                                "📋 Download Report as Markdown",
                                data=full_report_text,
                                file_name="research_report.md",
                                mime="text/markdown",
                            )
                        elif not final_data:
                            status.update(label="❌ No response", state="error")
                            st.error("No response received from the pipeline.")

                except requests.exceptions.RequestException as e:
                    status.update(label="❌ API Error", state="error")
                    st.error(f"API not reachable: {e}")

        else:
            # ── Non-streaming mode (original) ─────────────
            with st.status("Running multi-agent pipeline...", expanded=True) as status:
                st.write("🔍 Researcher Agent working...")
                st.write("📊 Analyst Agent working...")
                st.write("✍️ Writer Agent composing report...")

                try:
                    r = requests.post(
                        f"{api_url}/agent/chat",
                        json={"message": query, "thread_id": thread_id},
                        timeout=120,
                    )

                    if r.ok:
                        status.update(label="✅ Report ready!", state="complete")
                        data = r.json()
                        report = data["report"]

                        # Guardrail warning
                        if report["needs_human_review"]:
                            st.warning(
                                f"⚠️ Human review recommended — "
                                f"confidence: {report['confidence']:.2f}"
                            )

                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Confidence", f"{report['confidence']:.2f}")
                        col2.metric("Latency", f"{data['latency_ms']} ms")
                        col3.metric("Iterations", data["iterations"])

                        st.divider()

                        # Report sections
                        st.subheader(report["title"])
                        st.markdown(f"**Summary:** {report['summary']}")

                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown("**🔍 Key Findings**")
                            for f in report["research_findings"]:
                                if f:
                                    st.markdown(f"- {f}")

                        with col_b:
                            st.markdown("**📊 Analysis**")
                            for a in report["analysis"]:
                                if a:
                                    st.markdown(f"- {a}")

                        st.markdown("**Conclusion**")
                        st.write(report["conclusion"])

                        if report["sources"]:
                            with st.expander("📎 Sources"):
                                for s in report["sources"]:
                                    st.write(f"- {s}")

                    else:
                        status.update(label="❌ Failed", state="error")
                        st.error(f"Error: {r.text}")

                except requests.exceptions.RequestException as e:
                    status.update(label="❌ API Error", state="error")
                    st.error(f"API not reachable: {e}")

# ── Tab 2: Documents (RAG) ──────────────────────────────
with tabs[1]:
    st.subheader("📄 Document Upload & RAG Indexing")
    st.write("Upload PDF, TXT, or MD files to index them in the FAISS vector database. The Researcher Agent will automatically query these documents during research.")

    # 1. Check RAG Status from Backend
    rag_available = False
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=3)
        if health_resp.ok:
            rag_available = health_resp.json().get("rag_available", False)
    except Exception:
        st.warning("Could not contact the backend to verify RAG status.")

    if not rag_available:
        st.error("⚠️ **RAG Features Disabled:** The backend does not have FAISS / vector store dependencies installed. (Typically disabled on low-memory tiers like Render Free to avoid OOM crashes). To enable, deploy on a tier with >= 1GB RAM and install the packages listed in `requirements.txt`.")
    else:
        # 2. File Upload Form
        uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            if st.button("📤 Index Document", use_container_width=True):
                with st.spinner("Processing and indexing document..."):
                    try:
                        # Prepare files payload
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        upload_resp = requests.post(f"{api_url}/agent/upload", files=files, timeout=60)
                        
                        if upload_resp.ok:
                            st.success(f"Successfully indexed '{uploaded_file.name}'!")
                            st.json(upload_resp.json())
                        else:
                            st.error(f"Upload failed: {upload_resp.text}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        st.divider()
        
        # 3. Listed Documents
        st.subheader("📚 Indexed Documents")
        try:
            docs_resp = requests.get(f"{api_url}/agent/documents", timeout=5)
            if docs_resp.ok:
                docs_data = docs_resp.json()
                if docs_data.get("count", 0) == 0:
                    st.info("No documents uploaded yet.")
                else:
                    for doc in docs_data.get("documents", []):
                        st.markdown(f"- **{doc['filename']}** ({doc['text_length']} chars, {doc['num_chunks']} chunks)")
            else:
                st.error("Failed to load document list from backend.")
        except Exception as e:
            st.error(f"Could not load documents: {e}")

        st.divider()

        # 4. Search Test
        st.subheader("🔍 Vector Store Search Test")
        search_query = st.text_input("Enter a query to test search retrieval")
        if search_query:
            if st.button("Test Retrieve", use_container_width=True):
                with st.spinner("Searching vector database..."):
                    try:
                        search_resp = requests.post(f"{api_url}/agent/search?query={requests.utils.quote(search_query)}", timeout=5)
                        if search_resp.ok:
                            search_results = search_resp.json()
                            st.write(f"Found {search_results['count']} matches:")
                            for res in search_results["results"]:
                                with st.expander(f"Source: {res['source']} (Score: {res['score']})"):
                                    st.write(res["content"])
                        else:
                            st.error(f"Search failed: {search_resp.text}")
                    except Exception as e:
                        st.error(f"An error occurred during search: {e}")

# ── Tab 3: History ────────────────────────────────────────
with tabs[2]:
    st.subheader(f"Conversation History — Thread: {thread_id}")
    if st.button("Load History"):
        try:
            r = requests.get(
                f"{api_url}/agent/history/{thread_id}",
                timeout=5,
            )
            if r.ok:
                data = r.json()
                if data["count"] == 0:
                    st.info("No history for this thread yet.")
                else:
                    for i, ex in enumerate(data["exchanges"]):
                        with st.expander(f"Query {i+1}: {ex['query'][:60]}..."):
                            st.write(ex["report"])
        except Exception as e:
            st.error(str(e))

# ── Tab 4: About ──────────────────────────────────────────
with tabs[3]:
    st.subheader("System Architecture")
    st.markdown("""
    ### Agent Hierarchy
    | Agent | Role | Guardrail |
    |---|---|---|
    | **Researcher** | Gathers factual info via web search (tool-calling ReAct loop) | Flags [UNCERTAIN] content |
    | **Analyst** | Extracts insights; detects gaps for re-research | Flags [LOW-CONFIDENCE] |
    | **Writer** | Produces structured report via LLM structured output | Confidence threshold check |

    ### Key Features
    - LangGraph supervisor with conditional routing + self-reflection loop
    - Per-thread memory via SQLite + MemorySaver checkpointer
    - Confidence scoring on every agent output
    - Human review flag when confidence < threshold
    - Real-time SSE streaming of agent progress
    - Full agent error handling, timeout, and fallback
    - FastAPI backend with Pydantic validation
    - LangSmith tracing support
    """)
