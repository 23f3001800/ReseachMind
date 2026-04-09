import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
    thread_id = st.text_input("Thread ID (session)", value="default")

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
    Researcher Agent
        ↓
    Analyst Agent
        ↓
    Writer Agent
        ↓
    Structured Report
    ```
    """)

# ── Main ──────────────────────────────────────────────────
st.title("🤖 Agentic Research Assistant")
st.caption("Multi-agent system: Researcher → Analyst → Writer · Memory per session · Guardrails enabled")

tabs = st.tabs(["💬 Research", "📜 History", "📊 About"])

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
        else:
            with st.status("Running multi-agent pipeline...", expanded=True) as status:
                st.write("🔍 Researcher Agent working...")
                time.sleep(0.5)
                st.write("📊 Analyst Agent working...")
                time.sleep(0.5)
                st.write("✍️ Writer Agent composing report...")

                try:
                    r = requests.post(
                        f"{api_url}/agent/chat",
                        json={
                            "message": query,
                            "thread_id": thread_id,
                        },
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

# ── Tab 2: History ────────────────────────────────────────
with tabs[1]:
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

# ── Tab 3: About ──────────────────────────────────────────
with tabs[2]:
    st.subheader("System Architecture")
    st.markdown("""
    ### Agent Hierarchy
    | Agent | Role | Guardrail |
    |---|---|---|
    | **Researcher** | Gathers factual info via web search | Flags [UNCERTAIN] content |
    | **Analyst** | Extracts insights from research | Flags [LOW-CONFIDENCE] |
    | **Writer** | Produces structured report | Confidence threshold check |

    ### Key Features
    - LangGraph supervisor with conditional routing
    - Per-thread memory via MemorySaver checkpointer
    - Confidence scoring on every agent output
    - Human review flag when confidence < threshold
    - Full agent error handling and fallback
    - FastAPI backend with Pydantic validation
    """)