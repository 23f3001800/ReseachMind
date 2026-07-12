# 🤖 Autonomous AI Research Agent

A **multi-agent research system** that autonomously plans, searches the web, analyzes findings, and writes structured reports — all in under 60 seconds. Built with **LangGraph**, **Groq (LLaMA 3.3 70B)**, **DuckDuckGo Search**, **FastAPI**, and **Streamlit**.

> Give it a topic. Get back a structured research report with findings, analysis, and sources — complete with confidence scoring and human review flags.

---

## ✨ Features

- **Multi-Agent Pipeline** — Three specialized agents (Researcher → Analyst → Writer) collaborate through a LangGraph state graph
- **Web Search Integration** — Real-time information gathering via DuckDuckGo Search
- **Confidence Scoring** — Every agent output carries a confidence score (0.0 – 1.0)
- **Human Review Guardrails** — Automatically flags low-confidence reports for human review
- **Conditional Routing** — If the researcher flags uncertainty, the analyst is skipped and the writer produces a best-effort report
- **Conversation Memory** — Per-thread memory via LangGraph's `MemorySaver` checkpointer (up to 20 exchanges per thread)
- **Structured Output** — Reports are parsed into title, summary, key findings, analysis, conclusion, and sources
- **FastAPI Backend** — Fully documented REST API with Pydantic validation
- **Streamlit Frontend** — Interactive UI with real-time agent pipeline status, metrics, and conversation history
- **Docker Ready** — Dockerfile included for containerized deployment

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Research │  │  History   │  │  About   │  │  Settings   │ │
│  │   Tab    │  │   Tab      │  │   Tab    │  │  Sidebar    │ │
│  └────┬─────┘  └─────┬─────┘  └──────────┘  └─────────────┘ │
│       │               │                                      │
└───────┼───────────────┼──────────────────────────────────────┘
        │  HTTP         │  HTTP
        ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                            │
│                                                              │
│  POST /agent/chat ──────────────────────────────────┐        │
│  GET  /agent/history/{thread_id}                    │        │
│  DELETE /agent/history/{thread_id}                  │        │
│  GET  /agent/graph                                  │        │
│  GET  /health                                       │        │
│                                                     ▼        │
│  ┌───────────────── LangGraph Pipeline ──────────────────┐   │
│  │                                                       │   │
│  │   ┌────────────┐     ┌──────────┐     ┌────────────┐  │   │
│  │   │ Researcher │────▶│ Analyst  │────▶│   Writer   │  │   │
│  │   │   Agent    │     │  Agent   │     │   Agent    │  │   │
│  │   └──────┬─────┘     └──────────┘     └────────────┘  │   │
│  │          │                                             │   │
│  │          │  (low confidence)                           │   │
│  │          └───────────────────────────▶ Writer (skip)   │   │
│  │                                                       │   │
│  └───────────────────────────────────────────────────────┘   │
│                          │                                    │
│               ┌──────────┴──────────┐                        │
│               │   MemorySaver       │                        │
│               │  (per thread_id)    │                        │
│               └─────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────┐    ┌───────────┐
  │   Groq    │    │ DuckDuck  │
  │ LLaMA 3.3│    │ Go Search │
  │   70B     │    │           │
  └───────────┘    └───────────┘
```

### Agent Roles

| Agent | Responsibility | Guardrail |
|---|---|---|
| **Researcher** | Gathers factual info via web search; outputs numbered findings with sources | Flags `[UNCERTAIN]` content → drops confidence to 0.5 |
| **Analyst** | Extracts patterns, trends, insights, and gaps from research findings | Flags `[LOW-CONFIDENCE]` → caps confidence at 0.5 |
| **Writer** | Synthesizes research + analysis into a structured final report | Checks confidence against threshold (default 0.7); flags for human review if below |

---

## 📁 Project Structure

```
Autonomous-AI-Research-Agent/
├── backend/
│   ├── main.py                 # FastAPI app — endpoints, report parsing, lifespan
│   ├── config.py               # Pydantic Settings — env vars & defaults
│   ├── Dockerfile              # Container config (port 7860 for HuggingFace Spaces)
│   ├── requirements.txt        # Backend Python dependencies
│   ├── agents/
│   │   ├── researcher.py       # Researcher agent — web search + LLM findings
│   │   ├── analyst.py          # Analyst agent — insight extraction from research
│   │   └── writer.py           # Writer agent — structured report generation
│   ├── core/
│   │   ├── state.py            # AgentState TypedDict — shared graph state schema
│   │   ├── supervisor.py       # LangGraph StateGraph — build, compile, run pipeline
│   │   └── memory.py           # MemorySaver checkpointer + conversation store
│   └── schemas/
│       └── models.py           # Pydantic models — request/response/report schemas
├── frontend/
│   └── app.py                  # Streamlit UI — research tab, history, about
├── .env.example                # Template for required environment variables
├── Procfile                    # Heroku/Railway deployment config
├── runtime.txt                 # Python version (3.11.0)
├── requirements.txt            # Full frozen dependencies
└── LICENSE                     # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier available)
- *(Optional)* A [Tavily API key](https://tavily.com/) for enhanced search

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Autonomous-AI-Research-Agent.git
cd Autonomous-AI-Research-Agent
```

### 2. Set Up Environment Variables

```bash
cp .env.example backend/.env
```

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here    # optional
LLM_MODEL=llama-3.3-70b-versatile      # or any Groq-supported model
```

### 3. Create a Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 4. Run the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### 5. Run the Frontend

In a separate terminal:

```bash
cd frontend
streamlit run app.py
```

The Streamlit app will open at `http://localhost:8501`.

---

## 📡 API Reference

### `GET /health`

Health check.

```json
{ "status": "ok", "service": "agentic-research-assistant" }
```

### `POST /agent/chat`

Run the full multi-agent research pipeline.

**Request Body:**

```json
{
  "message": "What are the latest advances in multi-agent AI systems?",
  "thread_id": "default",
  "stream": false
}
```

**Response:**

```json
{
  "thread_id": "default",
  "report": {
    "title": "Research Report",
    "summary": "...",
    "research_findings": ["...", "..."],
    "analysis": ["...", "..."],
    "conclusion": "...",
    "sources": ["...", "..."],
    "confidence": 0.8,
    "needs_human_review": false
  },
  "latency_ms": 12345.67,
  "iterations": 3
}
```

### `GET /agent/history/{thread_id}`

Retrieve conversation history for a session.

### `DELETE /agent/history/{thread_id}`

Clear all memory for a session.

### `GET /agent/graph`

Returns the agent graph structure and routing logic.

---

## 🐳 Docker Deployment

```bash
cd backend
docker build -t research-agent .
docker run -p 7860:7860 --env-file .env research-agent
```

The API will be accessible at `http://localhost:7860`.

> **Note:** The Dockerfile exposes port `7860` for compatibility with HuggingFace Spaces.

---

## ⚙️ Configuration

All configuration is managed via environment variables (loaded from `.env`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | API key for Groq LLM inference |
| `TAVILY_API_KEY` | `""` | API key for Tavily search (optional) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |
| `TEMPERATURE` | `0.1` | LLM temperature for research/analysis |
| `MAX_ITERATIONS` | `10` | Maximum agent iterations |
| `CONFIDENCE_THRESHOLD` | `0.7` | Below this, reports are flagged for human review |

---

## 🛡️ Guardrails

The system implements multiple layers of safety:

1. **Researcher** — Flags `[UNCERTAIN]` content, reducing confidence to 0.5
2. **Analyst** — Flags `[LOW-CONFIDENCE]` analysis, capping confidence at 0.5
3. **Writer** — Compares final confidence against `CONFIDENCE_THRESHOLD`; flags reports below it for human review
4. **Conditional Routing** — If the researcher encounters errors or high uncertainty, the analyst is skipped entirely
5. **Error Fallback** — Every agent catches exceptions and produces a degraded output with `needs_human_review = True`
6. **Input Validation** — Pydantic enforces message length (3–1000 chars) and type constraints

---

## 🗺️ Roadmap

- [ ] Streaming responses via WebSocket / SSE
- [ ] Persistent memory with SQLite or PostgreSQL
- [ ] Tavily search integration as primary search tool
- [ ] Copy report to clipboard button in the UI
- [ ] Demo GIF and architecture diagram images
- [ ] Deploy to Streamlit Cloud / HuggingFace Spaces
- [ ] Evaluation framework for report quality

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — LLaMA 3.3 70B Versatile |
| **Agent Framework** | LangGraph + LangChain |
| **Web Search** | DuckDuckGo Search (via `langchain-community`) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker |
| **Language** | Python 3.11 |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Vikas Rajput**

---

<p align="center">
  Built with ❤️ using LangGraph, Groq, and FastAPI
</p>