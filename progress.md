# Progress

Gap IDs (G-n) refer to [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Setup
- [x] Repo created
- [x] Base folders created
- [x] Requirements added (core + optional RAG split)
- [x] README written
- [x] Dependency versions pinned (requirements.lock)

## Backend
- [x] FastAPI app running — 12 endpoints
- [x] LangGraph supervisor: Researcher → Analyst → Writer with conditional routing
- [x] Pydantic schemas + validation
- [x] Error handling, fallbacks, and global exception handler
- [x] Citations restricted to retrieved pages (G-1)
- [x] Measured token/cost accounting from provider metadata (G-2)
- [x] Retry pass targets the analyst's identified gaps (G-3)
- [x] Gap detection counts items, ignores "none" answers (G-5)
- [x] Streaming path records history and usage (G-4)
- [x] API key auth, CORS allowlist, per-client rate limiting (G-11)
- [x] Upload hardening: generated filenames, size cap (G-12)
- [x] Timeouts and loop caps moved to settings (G-19)
- [x] Dependencies pinned in requirements.lock (G-9)
- [x] Persistent storage — vector index, catalog and history survive restart (G-6, G-7)
- [x] Structured JSON logging for Log Analytics (G-17)
- [x] RAG enabled by default via fastembed instead of torch (G-8)
- [x] Tavily fixed — it reads TAVILY_API_KEY from env, not an api_key kwarg
- [x] Malformed provider tool calls degrade instead of losing the whole pass
- [x] In-flight progress events (agent_start / tool_call / tool_result)
- [x] Golden-set evaluation harness (backend/eval)
- [ ] Chunk offset calculation rewritten (G-16)
- [ ] Rate limiter moved off per-instance memory (only matters above 1 replica)

## Frontend — React 18 + TypeScript (Streamlit removed)
- [x] Vite + React + strict TypeScript, no UI-library dependency
- [x] Design-token system with light / dark / system themes
- [x] Typed API client with SSE reader and AbortController support
- [x] Live agent-pipeline visualisation with retry indication
- [x] Sources render as clickable links with provider attribution
- [x] Per-session thread IDs — visitors no longer share one history
- [x] Evaluate view (LLM-as-judge, 4 dimensions)
- [x] Usage/cost view
- [x] Cancel button for in-flight runs
- [x] Responsive layout, keyboard focus, aria-live, reduced-motion
- [x] Live activity feed — search queries shown as the researcher runs them
- [x] Pipeline state driven by real agent_start events, not inferred
- [ ] Per-token streaming of the writer's prose (structured output makes this awkward)
- [ ] Report library (persisted, searchable past reports)

## Production signals
- [x] Evaluation framework (4-dimension LLM-as-judge)
- [x] Guardrails: confidence scoring, review flags, timeouts, retry cap
- [x] Logging + optional LangSmith tracing
- [x] Deployed on Azure Container Apps — managed identity, scale-to-zero, $PORT
- [x] API key never reaches the browser (nginx injects it server-side)
- [x] Reproducible deploy script (deploy/azure.sh)
- [x] Docker build validation in CI
- [x] Golden-set eval harness (run manually or on a schedule — it costs real LLM calls)
- [ ] Coverage gate in CI

## Polish
- [x] README polished
- [x] Architecture diagram (ASCII)
- [ ] Demo screenshots/GIF
