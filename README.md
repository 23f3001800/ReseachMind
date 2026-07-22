<div align="center">

# 🤖 researchMind

<h3>A production-grade multi-agent research system powered by LangGraph, FastAPI & React</h3>

Three specialized AI agents — **Researcher**, **Analyst**, and **Writer** — collaborate through a supervised pipeline with self-reflection, guardrails, and real-time streaming. Every citation is a page the system actually retrieved.

[![CI](https://img.shields.io/github/actions/workflow/status/23f3001800/Autonomous-AI-Research-Agent/ci.yml?branch=main&style=for-the-badge&logo=github-actions&label=CI)](https://github.com/23f3001800/Autonomous-AI-Research-Agent/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://console.groq.com)
[![React](https://img.shields.io/badge/React_18-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![Azure](https://img.shields.io/badge/Azure-Container_Apps-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/products/container-apps)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

**[Features](#-key-features)** · **[Architecture](#%EF%B8%8F-architecture)** · **[Quick Start](#-quick-start)** · **[API Docs](#-api-endpoints)** · **[Testing](#-testing)** · **[Evaluation](#-evaluation)**

</div>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔄 **Multi-Agent Pipeline** | Researcher → Analyst → Writer with LangGraph supervisor and conditional routing |
| 🛠️ **ReAct Tool Calling** | Researcher LLM autonomously decides when and what to search via tool-calling loop |
| 🔁 **Self-Reflection Loop** | Analyst detects research gaps → routes back to Researcher for a second pass |
| 📄 **RAG Pipeline** | Upload PDF/TXT/MD → chunk → FAISS vector search → inject context into agents |
| 🔗 **Traceable Citations** | Sources come only from pages the system actually retrieved — the model can't invent one |
| ⚡ **SSE Streaming** | Real-time Server-Sent Events show each agent's progress as it completes |
| 🎨 **React UI** | TypeScript SPA with a live pipeline view, light/dark themes, and no UI-library dependencies |
| ⚖️ **LLM-as-Judge** | 4-dimension evaluation framework (Accuracy, Depth, Completeness, Clarity) |
| 🧩 **Structured Output** | Writer uses Pydantic-validated `.with_structured_output()` — no regex parsing |
| 🛡️ **Guardrails** | Confidence scoring, human review flags, error fallback, request timeout (120s) |
| 💾 **Persistent Memory** | SQLite-backed conversation history per thread + LangGraph checkpointer |
| 💰 **Token/Cost Tracking** | Per-request and cumulative token usage with Groq pricing estimates |
| 🔍 **LangSmith Tracing** | Auto-enabled when `LANGSMITH_API_KEY` is set |
| 🔎 **Configurable Search** | Auto-selects Tavily (quality) or DuckDuckGo (free) based on API key |
| 🤖 **GitHub Actions CI** | Automated linting (Ruff) and testing (pytest) on every push/PR |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           React 18 + TypeScript SPA (Vite)           │
│   live pipeline view · SSE streaming · light/dark    │
└──────────────────────┬──────────────────────────────┘
                       │ same-origin /api/*
┌──────────────────────▼──────────────────────────────┐
│                    nginx (sidecar)                   │
│   serves the SPA · proxies /api/ · injects the API   │
│   key server-side so the browser never holds one     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS + X-API-Key
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                     │
│  /agent/chat  /agent/chat/stream  /agent/evaluate   │
│  /agent/upload  /agent/search  /agent/usage         │
│  API-key auth · rate limiting · CORS allowlist      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              LangGraph Supervisor                    │
│                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │Researcher│───▶│ Analyst  │───▶│  Writer  │      │
│   │(ReAct +  │    │(gap det.)│    │(struct.  │      │
│   │ tools)   │◀───│          │    │ output)  │      │
│   └──────────┘    └──────────┘    └──────────┘      │
│        │ retry loop                                  │
│                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │  FAISS   │  │  SQLite  │  │  LangSmith       │  │
│   │ VectorDB │  │  Memory  │  │  Tracing         │  │
│   └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Autonomous-AI-Research-Agent/
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline
├── backend/
│   ├── agents/
│   │   ├── researcher.py            # ReAct agent with tool-calling loop
│   │   ├── analyst.py               # Insight extraction + gap detection
│   │   ├── writer.py                # Structured report via .with_structured_output()
│   │   └── tools.py                 # Web search tools (DuckDuckGo / Tavily)
│   ├── core/
│   │   ├── supervisor.py            # LangGraph orchestration + streaming
│   │   ├── state.py                 # AgentState TypedDict
│   │   ├── memory.py                # SQLite conversation history
│   │   ├── logger.py                # Structured logging factory
│   │   ├── rag.py                   # Document loading + chunking
│   │   ├── vectorstore.py           # FAISS vector search
│   │   ├── evaluator.py             # LLM-as-judge evaluation
│   │   └── usage.py                 # Token/cost tracking
│   ├── schemas/models.py            # Pydantic models (request/response/agent)
│   ├── config.py                    # Settings with env var binding
│   ├── main.py                      # FastAPI app (12 endpoints)
│   ├── tests/                       # 38 unit + integration tests
│   ├── requirements.txt             # core deps
│   ├── requirements-rag.txt         # optional vector-search stack (~1GB RAM)
│   └── Dockerfile
├── frontend/                        # React 18 + TypeScript + Vite
│   ├── src/
│   │   ├── api/client.ts            # typed client, SSE reader, abort support
│   │   ├── components/              # ui primitives, Pipeline, ReportView
│   │   ├── views/                   # Research, Documents, Evaluate, Usage, History
│   │   ├── hooks/                   # persisted settings + theme
│   │   ├── styles/                  # design tokens, light/dark themes
│   │   └── App.tsx
│   ├── nginx.conf.template          # SPA + /api proxy with key injection
│   ├── docker-entrypoint.sh         # renders nginx.conf from env at start
│   └── Dockerfile
├── deploy/azure.sh                  # reproducible Azure Container Apps deploy
├── Dockerfile                       # all-in-one image for local use
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/23f3001800/Autonomous-AI-Research-Agent.git
cd Autonomous-AI-Research-Agent

# Create .env from template
cp .env.example backend/.env
```

### 2. Add your API key

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_key_here          # Required — get free at console.groq.com
TAVILY_API_KEY=your_tavily_key_here      # Optional — better search quality
LANGSMITH_API_KEY=your_langsmith_key     # Optional — enables tracing
SEARCH_PROVIDER=auto                      # auto | tavily | duckduckgo
```

### 3. Install & Run

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend — in a second terminal (Node 20+)
cd frontend
npm install
npm run dev
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`, so the SPA talks to the same
origin in development and in production. Point it elsewhere with
`VITE_DEV_API=http://host:port npm run dev`.

### 4. Docker

```bash
# Both services in one container (local convenience)
docker build -t research-agent .
docker run -p 8080:8080 --env-file backend/.env research-agent

# Or build the two service images used in deployment
docker build -t research-agent-api ./backend
docker build -t research-agent-ui  ./frontend
```

> **Ports:** UI at `http://localhost:8080` · API (direct) at `http://localhost:8000` · Swagger docs at `http://localhost:8000/docs`

### 5. Deploy to Azure

```bash
./deploy/azure.sh --groq-key <YOUR_GROQ_KEY>
```

Provisions a resource group, ACR, a Container Apps environment and both apps —
managed-identity registry access, a generated API key, and CORS locked to the UI
origin. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) §6 for cost shape and operations.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/agent/chat` | Run full pipeline (sync) |
| `POST` | `/agent/chat/stream` | Run pipeline with SSE streaming |
| `POST` | `/agent/evaluate` | Run pipeline + LLM-as-judge scoring |
| `GET` | `/agent/history/{id}` | Get conversation history |
| `DELETE` | `/agent/history/{id}` | Clear thread memory |
| `GET` | `/agent/graph` | Show agent graph structure |
| `POST` | `/agent/upload` | Upload document for RAG |
| `GET` | `/agent/documents` | List uploaded documents |
| `POST` | `/agent/search` | Semantic search over documents |
| `GET` | `/agent/usage` | Token/cost analytics |
| `DELETE` | `/agent/usage` | Reset usage counters |

> 📖 **Interactive API docs:** `http://localhost:8000/docs` (Swagger UI)

---

## 🧪 Testing

```bash
# Backend — 38 tests
cd backend
pytest tests/ -v
ruff check . --ignore E501

# Frontend — strict TypeScript, no `any`
cd frontend
npm run typecheck
npm run build
```

---

## 🔧 Configuration

All settings are configurable via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *required* | Groq API key for LLM inference |
| `TAVILY_API_KEY` | `""` | Tavily API key for premium search |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `SEARCH_PROVIDER` | `auto` | `auto` / `tavily` / `duckduckgo` |
| `CONFIDENCE_THRESHOLD` | `0.7` | Below this → human review flag |
| `DB_PATH` | `data/memory.db` | SQLite database location |
| `REQUEST_TIMEOUT` | `180` | Seconds before the pipeline is abandoned |
| `MAX_TOOL_ROUNDS` | `5` | Researcher ReAct loop cap |
| `MAX_RESEARCH_RETRIES` | `1` | Self-reflection passes back to the researcher |
| `API_KEY` | `""` | Required in `X-API-Key` when set. **Set this before exposing the API** |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-client sliding window; `0` disables |
| `MAX_UPLOAD_MB` | `10` | Upload size cap |
| `LANGSMITH_API_KEY` | `""` | Enables LangSmith tracing |
| `LANGCHAIN_PROJECT` | `researchmind` | LangSmith project name |

The frontend container takes two of its own:

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | Upstream the nginx `/api/` proxy targets |
| `BACKEND_API_KEY` | `""` | Injected as `X-API-Key` server-side; never reaches the browser |

---

## 🛡️ Guardrails

| # | Guardrail | How it works |
|:---:|---|---|
| 1 | **Confidence Scoring** | Every agent outputs a confidence score (0-1). Below threshold triggers human review flag |
| 2 | **Self-Reflection** | Analyst detects significant research gaps → routes back to Researcher (max 1 retry) |
| 3 | **Error Fallback** | Agent failures produce degraded but usable output instead of crashing |
| 4 | **Request Timeout** | Configurable hard limit (default 180s) prevents infinite LLM loops |
| 5 | **Input Validation** | Pydantic enforces message length (3-1000 chars) and request schema |
| 6 | **Global Exception Handler** | Unhandled errors return structured JSON, not stack traces |
| 7 | **Traceable Citations** | Sources come only from executed searches — the model cannot invent one |
| 8 | **API Key + Rate Limit** | `X-API-Key` enforced, sliding-window per-client limit, CORS allowlist |
| 9 | **Upload Safety** | Generated filenames, size cap, extension allowlist |

---

## 📊 Evaluation

The built-in LLM-as-judge evaluator scores reports on 4 weighted dimensions:

| Dimension | Weight | What it measures |
|---|:---:|---|
| **Factual Accuracy** | 30% | Are findings verifiable and correct? |
| **Analytical Depth** | 25% | Are insights meaningful and nuanced? |
| **Completeness** | 25% | Does the report cover the query thoroughly? |
| **Clarity** | 20% | Is the writing clear and professional? |

```bash
# Run evaluation via API
curl -X POST http://localhost:8000/agent/evaluate \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest trends in AI agents?"}'
```

---

## 🧠 Tech Stack

<div align="center">

| Layer | Technology |
|:---:|---|
| **Orchestration** | LangGraph (StateGraph with conditional routing) |
| **LLM** | Groq (Llama 3.3 70B) with tool calling |
| **Search** | DuckDuckGo (free) / Tavily (premium) |
| **RAG** | FAISS + HuggingFace `all-MiniLM-L6-v2` |
| **Backend** | FastAPI (async, SSE streaming) |
| **Frontend** | React 18 + TypeScript + Vite · zero UI dependencies, custom design system |
| **Edge** | nginx — SPA hosting, API proxy, server-side key injection |
| **Hosting** | Azure Container Apps (scale-to-zero) + ACR |
| **Memory** | SQLite + LangGraph MemorySaver |
| **Tracing** | LangSmith (optional) |
| **CI/CD** | GitHub Actions (Ruff + pytest) |
| **Container** | Docker |

</div>

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

<h4>Built with ❤️ by <a href="https://github.com/23f3001800">23f3001800</a></h4>

⭐ Star this repo if you found it useful!

</div>
