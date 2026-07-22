<div align="center">

# 🤖 researchMind

<h3>A multi-agent research system built on LangGraph, FastAPI and React</h3>

Three specialized agents — **Researcher**, **Analyst** and **Writer** — collaborate through a supervised pipeline with self-reflection, guardrails and live streaming. **Every citation is a page the system actually retrieved**, never one the model invented.

[![CI](https://img.shields.io/github/actions/workflow/status/23f3001800/Autonomous-AI-Research-Agent/ci.yml?branch=main&style=flat-square&logo=github&label=CI)](https://github.com/23f3001800/Autonomous-AI-Research-Agent/actions)
[![Tests](https://img.shields.io/badge/tests-51_passing-2f7a4d?style=flat-square)](#-testing)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React_18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Azure](https://img.shields.io/badge/Azure-Container_Apps-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/products/container-apps)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

### [→ Try it live](https://research-agent-ui.agreeablestone-a7a39990.centralindia.azurecontainerapps.io)

<sub>Hosted on Azure Container Apps. It scales to zero when idle, so the **first request after a quiet period takes up to a minute** to wake the container — after that it responds normally.</sub>

---

**[Features](#-key-features)** · **[Architecture](#%EF%B8%8F-architecture)** · **[Quick Start](#-quick-start)** · **[API](#-api-endpoints)** · **[Testing](#-testing)** · **[Evaluation](#-evaluation)**

</div>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔗 **Traceable citations** | Sources are collected from *executed* searches only. A report can cite nothing the system didn't fetch |
| 🔄 **Multi-agent pipeline** | Researcher → Analyst → Writer on a LangGraph supervisor with conditional routing |
| 🛠️ **ReAct tool calling** | The researcher decides when and what to search, across multiple rounds |
| 🔁 **Gap-targeted retry** | The analyst's identified gaps are fed back into a second research pass — not a repeat of the first |
| 📡 **Live progress** | SSE streams `agent_start` / `tool_call` / `tool_result` mid-run, so you watch each search as it happens |
| 📄 **RAG over your documents** | Upload PDF/TXT/MD → chunk → vector search → injected into the researcher alongside web results |
| 💾 **Survives restarts** | Vector index, document catalog and history persist to a mounted volume |
| 🎨 **React UI** | TypeScript SPA, light/dark, cancellable runs, zero UI-library dependencies (56 kB gzipped) |
| 🔐 **Key never in the browser** | nginx proxies `/api` and injects `X-API-Key` server-side; same-origin, so no CORS surface |
| ⚖️ **LLM-as-judge** | 4-dimension evaluation plus a golden-set harness that fails on quality regression |
| 🧩 **Structured output** | Writer uses Pydantic-validated `.with_structured_output()` — no regex parsing |
| 💰 **Measured token/cost** | Read from the provider's `usage_metadata`, per agent — measured, not estimated |
| 🛡️ **Guardrails** | Confidence scoring, human-review flags, error fallback, timeouts, rate limiting, upload hardening |
| 📋 **JSON logging** | `LOG_FORMAT=json` emits queryable fields for Log Analytics |
| 🔍 **LangSmith tracing** | Auto-enabled when `LANGSMITH_API_KEY` is set |
| 🤖 **CI** | Ruff + pytest, frontend typecheck + build, and Docker image builds for both services |

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

## 🔗 How citations stay honest

The usual failure of an LLM research tool is a confident report citing sources
that don't exist. researchMind makes that structurally impossible rather than
discouraging it in a prompt:

1. **Search tools record what they fetch.** Each tool writes `{url, title, provider}`
   into a per-request collector as results come back.
2. **The report's sources come from that collector**, never from the model's own
   text. If the model writes a plausible-looking citation in its prose, it is not
   in the collector, so it is not in the report.
3. **No search means no citations.** If every search fails, the report ships with
   an empty source list and a visible warning — not a fabricated bibliography.

The evaluation harness enforces this too: a run **fails** if any report cites
nothing, regardless of how well it scores on quality.

---

## 📁 Project Structure

```
researchMind/
├── .github/workflows/ci.yml         # 3 jobs: backend, frontend, image builds
├── backend/
│   ├── agents/
│   │   ├── researcher.py            # ReAct loop, gap-targeted retry, tool-failure recovery
│   │   ├── analyst.py               # Insight extraction + gap detection
│   │   ├── writer.py                # Structured report via .with_structured_output()
│   │   └── tools.py                 # Search tools + the source collector
│   ├── core/
│   │   ├── supervisor.py            # LangGraph orchestration + streaming
│   │   ├── state.py                 # AgentState TypedDict
│   │   ├── events.py                # In-flight progress sink (tool calls, agent starts)
│   │   ├── memory.py                # SQLite conversation history
│   │   ├── logger.py                # Text or JSON logging
│   │   ├── rag.py                   # Document loading + chunking
│   │   ├── vectorstore.py           # fastembed + FAISS, persisted to disk
│   │   ├── evaluator.py             # LLM-as-judge evaluation
│   │   └── usage.py                 # Measured token/cost tracking
│   ├── eval/                        # Golden-set regression harness
│   ├── schemas/models.py            # Pydantic request/response models
│   ├── config.py                    # Settings with env var binding
│   ├── main.py                      # FastAPI app (12 endpoints)
│   ├── tests/                       # 51 unit + integration tests
│   ├── requirements.lock            # Pinned — what actually ships
│   ├── requirements-rag.txt         # Vector stack (fastembed, faiss, pypdf)
│   └── Dockerfile
├── frontend/                        # React 18 + TypeScript + Vite
│   ├── src/
│   │   ├── api/client.ts            # Typed client, SSE reader, abort support
│   │   ├── components/              # ui primitives, Pipeline, ActivityFeed, ReportView
│   │   ├── views/                   # Research, Documents, Evaluate, Usage, History
│   │   ├── hooks/                   # Persisted settings + theme
│   │   ├── styles/                  # Design tokens, light/dark themes
│   │   └── App.tsx
│   ├── nginx.conf.template          # SPA + /api proxy with server-side key injection
│   ├── docker-entrypoint.sh         # Renders nginx.conf from env at start
│   └── Dockerfile
├── deploy/azure.sh                  # Reproducible Azure Container Apps deploy
├── Dockerfile                       # All-in-one image for local use
├── PROJECT_GUIDE.md                 # Architecture review, gap analysis, operations
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/23f3001800/ResearchMind.git
cd ResearchMind

# Create .env from template
cp .env.example backend/.env
```

### 2. Add your API key

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_key_here          # Required — free at console.groq.com
TAVILY_API_KEY=your_tavily_key_here      # Optional — better search quality
LANGSMITH_API_KEY=your_langsmith_key     # Optional — enables tracing
SEARCH_PROVIDER=auto                      # auto | tavily | duckduckgo
```

### 3. Install & Run

```bash
# Backend — the lockfile is what CI and the images build from
cd backend
pip install -r requirements.lock

# Optional: document upload and semantic search (~250 MB of ONNX runtime)
pip install -r requirements-rag.txt

uvicorn main:app --reload --port 8000

# Frontend — in a second terminal (Node 20+)
cd frontend
npm install
npm run dev
```

Without the RAG extras the API still runs — `/health` reports
`"rag_available": false` and the document endpoints return 501 rather than
failing at first use.

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

Provisions a resource group, ACR, a Container Apps environment, an Azure Files
share for persistent state, and both apps — with managed-identity registry
access, a generated API key, and CORS locked to the UI origin. Roughly **$5–8 a
month** at low traffic, since both apps scale to zero.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the architecture review, the
outstanding gaps, and operational notes (key rotation, cost shape, and the
platform quirks worth knowing before you deploy).

> **Before exposing any deployment publicly, set `API_KEY`.** Without it,
> anyone who finds the URL can spend your Groq quota.

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

All endpoints except `/health` and the docs require an `X-API-Key` header when
`API_KEY` is set, and are rate limited per client.

> 📖 **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)
> 🌐 **Live API:** [`/api/health`](https://research-agent-ui.agreeablestone-a7a39990.centralindia.azurecontainerapps.io/api/health) — proxied through the UI, which supplies the key

---

## 🧪 Testing

```bash
# Backend — 51 tests, no network or API key needed
cd backend
pytest tests/ -q
ruff check . --ignore E501

# Frontend — strict TypeScript
cd frontend
npm run typecheck
npm run build
```

The suite is fully offline: LLM and search calls are mocked, so it runs in ~3
seconds and costs nothing. For a check against the real providers, use the
evaluation harness below.

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
| `DATA_DIR` | `data` | Storage root — vector index, catalog, history. **Mount a volume here in production** |
| `DB_PATH` | `{DATA_DIR}/memory.db` | Override the SQLite path specifically |
| `LOG_FORMAT` | `text` | `json` for structured logs a log aggregator can query |
| `LOG_LEVEL` | `INFO` | Standard Python levels |
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
| 2 | **Self-Reflection** | Analyst detects significant research gaps → routes back to Researcher (max 1 retry), carrying the gaps into the retry prompt |
| 3 | **Error Fallback** | Agent failures produce degraded but usable output instead of crashing. A malformed provider tool call falls back to writing up sources already gathered rather than losing the pass |
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

Grade a single query through the API:

```bash
curl -X POST http://localhost:8000/agent/evaluate \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest trends in AI agents?"}'
```

### Golden-set regression harness

A fixed set of queries scored by the judge, so quality drift is measurable
rather than anecdotal. It **fails the run if the mean score drops below
threshold, or if any report cites nothing** — the failure mode this project
exists to prevent.

```bash
cd backend
python -m eval.run_eval               # full set
python -m eval.run_eval --limit 2     # quick check
python -m eval.run_eval --json out.json
```

Kept out of the unit suite deliberately: it makes real LLM and search calls,
takes minutes, and costs money. Run it before a release or on a schedule.

---

## 🧠 Tech Stack

<div align="center">

| Layer | Technology |
|:---:|---|
| **Orchestration** | LangGraph (StateGraph with conditional routing) |
| **LLM** | Groq (Llama 3.3 70B) with tool calling |
| **Search** | Tavily (quality) or DuckDuckGo via `ddgs` (free) |
| **RAG** | fastembed `BAAI/bge-small-en-v1.5` (ONNX) + FAISS, persisted to disk |
| **Backend** | FastAPI (async, SSE streaming) |
| **Frontend** | React 18 + TypeScript + Vite · custom design system, no UI library |
| **Edge** | nginx — SPA hosting, API proxy, server-side key injection |
| **Hosting** | Azure Container Apps (scale-to-zero) + ACR + Azure Files |
| **Memory** | SQLite on a mounted volume + LangGraph checkpointer |
| **Tracing** | LangSmith (optional) |
| **CI** | GitHub Actions — Ruff, pytest, tsc, Docker builds |

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

**[Live demo](https://research-agent-ui.agreeablestone-a7a39990.centralindia.azurecontainerapps.io)** · **[Project guide](PROJECT_GUIDE.md)**

<sub>Built by <a href="https://github.com/23f3001800">23f3001800</a></sub>

⭐ Star this repo if you found it useful

</div>
