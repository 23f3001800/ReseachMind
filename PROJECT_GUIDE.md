# Project Guide — researchMind

**Reviewed:** 2026-07-21 · base commit `7f13232`
**Status:** 38/38 backend tests passing, ruff clean, frontend builds clean under strict TS. Nothing committed.
**Frontend:** Streamlit removed — replaced by a React 18 + TypeScript SPA (§5).
**Deployed:** live on **Azure Container Apps** (§6). Two apps, managed-identity registry access, API-key auth enforced server-side.

---

## 1. What problem does this actually solve?

### The claimed problem

Desk research is a slow serial loop: *search → skim → cross-check → synthesize → write up*. A knowledge worker doing a "what's the current state of X?" report spends 30–90 minutes on it, and most of that is mechanical.

This project automates that loop with three specialized agents and returns a structured report with confidence scoring and a human-review flag.

### The honest problem statement

Before this pass the project solved: *"demonstrate a multi-agent LangGraph pipeline end-to-end."* It could not solve the research problem itself, because **the output was not verifiable** — and the root cause turned out to be worse than a parsing shortcut.

The `sources` list was scraped from the model's own prose after the string `SOURCES:`. Underneath that, **web search was failing on every single request**: `langchain-community` now imports the `ddgs` package, `requirements.txt` pinned the retired `duckduckgo-search` name, and the tool swallowed the resulting `ImportError` and returned *"Search failed… Rely on your internal knowledge."* The agent then dutifully invented a citation list. Verified directly against the old code path:

```
OLD PATH RAISED: ImportError  Could not import ddgs python package.
```

So the deployed system was a zero-retrieval LLM with a fabricated bibliography. That is now fixed (§3.1) and verified against live search.

### Who would actually use it

| Persona | Job to be done | Ready now? |
|---|---|---|
| Analyst / consultant | First-draft market or tech scan with citations | ✅ citations are now real and traceable |
| Founder / PM | "What's the landscape for X?" before a decision | ✅ for a first pass |
| Student / researcher | Summary fused over uploaded PDFs | ⚠️ RAG needs the optional deps and ~1 GB RAM |
| Engineer learning agents | Reference implementation of LangGraph + FastAPI | ✅ |

### The wedge worth pursuing

The differentiated feature is not "AI writes a report" — that is commodity. It is **RAG over your own documents fused with live web search in one pipeline**: *"Here is our internal strategy PDF; tell me what changed in the market since we wrote it, and cite both."* That path exists in skeleton form at [backend/agents/researcher.py:51-75](backend/agents/researcher.py#L51-L75) and is the highest-leverage thing to build next.

---

## 2. How the system works

```
React 18 + TypeScript SPA (frontend/src) — 5 views, live pipeline, light/dark
    │  same-origin /api/*  (no credential in the browser)
nginx sidecar — serves the SPA, proxies /api/, injects X-API-Key server-side
    │  HTTPS + X-API-Key
FastAPI (backend/main.py) — 12 endpoints, API key + rate limit + CORS allowlist
    │
LangGraph StateGraph (backend/core/supervisor.py)
    │
    ├─ researcher_node  ReAct loop, ≤5 tool rounds, DuckDuckGo|Tavily
    │       ↑ retry (max 1) — now carries the analyst's gap list
    ├─ analyst_node     insights + gap detection → routes back or forward
    └─ writer_node      .with_structured_output(WriterReport) → END
    │
    ├─ source collector  thread-local, populated only by executed searches
    ├─ usage collector   thread-local, reads provider usage_metadata
    ├─ FAISS             in-process (resets on restart)
    ├─ MemorySaver       in-process (resets on restart)
    └─ SQLite            data/memory.db
```

**What was already well built:** clean agent separation with real prompt discipline; every node degrades instead of crashing; `.with_structured_output()` rather than regex-parsing; `asyncio.to_thread` so the event loop isn't starved; graceful `RAG_AVAILABLE` degradation; a thread-safe usage tracker; real CI.

---

## 3. Gap analysis

### 3.1 Fixed in this pass ✅

**G-22 · Web search was dead on arrival** 🔴 → fixed
`langchain-community` 0.4.x imports `ddgs`; requirements pinned the retired `duckduckgo-search`. Every free-tier search raised `ImportError`, was caught, and returned "Rely on your internal knowledge." Swapped to `ddgs` in both requirements files. Verified live — 4 real URLs returned for a test query.

**G-1 · Sources were fabricated** 🔴 → fixed
Tools now record `{url, title, provider}` into a thread-local collector as they execute ([backend/agents/tools.py:36-56](backend/agents/tools.py#L36-L56)), and the researcher reads `state["sources"]` from that collector only ([backend/agents/researcher.py](backend/agents/researcher.py)). The `SOURCES:` text parsing is gone; the system prompt now demands real URLs and a `[UNVERIFIED]` marker for memory-derived claims. `FinalReport.sources` is now `List[Source]`.
*Verified end-to-end through the real graph:* the stub model emitted a citation to *"The Journal Of Things I Made Up, 2026"* and it did **not** appear in the report's sources; 7 genuine URLs did.

**G-2 · Token and cost numbers were invented** 🔴 → fixed
`iterations * 500` is gone. `UsageCallbackHandler` ([backend/core/usage.py](backend/core/usage.py)) reads `usage_metadata` off every response and is attached to all four LLM constructors (researcher, analyst, writer, evaluator). Aggregated per agent and returned on `/agent/chat`, the streaming `complete` event, and `/agent/evaluate`.
*Verified:* per-agent breakdown `{researcher: 600, analyst: 560, writer: 450}` on a run with a known token budget.

**G-3 · The self-reflection loop was a no-op** 🔴 → fixed
The analyst now publishes `research_gaps_detail`, and the researcher builds a distinct second-pass prompt that names the gaps and passes the prior findings as context.
*Verified:* the retry prompt contains `CLOSE THOSE GAPS` plus the literal gap text.

**G-5 · Gap detection was a line count** 🟠 → fixed
`_extract_gaps` ([backend/agents/analyst.py](backend/agents/analyst.py)) counts bullet/numbered *items*, folds wrapped continuation lines into the item above, and drops "None significant" style non-answers.
*Caught by its own test during this pass:* my first regex used a bare `no\s` prefix, which silently ate legitimate gaps like *"No regional breakdown"*. Now anchored end-to-end so only a whole-item non-answer matches.

**G-4 · Streaming path never saved history** 🟠 → fixed
`/agent/chat/stream` now records history and usage in its `complete` branch. Streaming is the UI default, so this was the common path.

**G-11 · No authentication, CORS wide open** 🔴 → fixed
Middleware enforcing `X-API-Key` plus a sliding-window per-client rate limit; CORS driven by `ALLOWED_ORIGINS`. `/health` and the docs stay public. Empty `API_KEY` leaves it open for local dev and logs a startup warning.
*Verified:* missing key → 401, wrong key → 401, correct key → 200, 4th request in a 3/min window → 429.

**G-12 · Path traversal on upload** 🔴 → fixed
Filenames are stripped to their basename for display and written under a generated UUID, with a size cap (`MAX_UPLOAD_MB`) and empty-file rejection; the raw file is deleted after indexing.
*Verified:* a `../../../etc/passwd` upload is rejected.

**G-13 · SSE bridge busy-waited** 🟠 → fixed
The `while q.empty(): await asyncio.sleep(0.1)` poll is replaced by `loop.call_soon_threadsafe` into an `asyncio.Queue`. The worker thread is retained deliberately — the collectors are thread-local, and `astream()` would scatter sync nodes across executor threads and break accumulation.

**G-14/G-15/G-19/G-20** 🟢 → fixed
Dead `route_after_supervisor` removed · `/agent/search` takes a Pydantic body · timeouts and loop caps moved to `Settings` (`REQUEST_TIMEOUT` now 180s) · `progress.md` regenerated to reflect reality · unused `sseclient` import and dependency dropped · `class Config` migrated to `SettingsConfigDict`.

**G-8 · RAG shipped disabled** 🟠 → improved
Split into [backend/requirements-rag.txt](backend/requirements-rag.txt) with an explanation of the RAM cost. The frontend's hardcoded Render apology is replaced by a message driven off `/health`.

**G-10 · Dockerfiles not deployable** 🟠 → improved
No more `.env.example` → `.env` baking (so a missing key fails fast at startup, not at first LLM call); non-root user; multi-stage backend build; `$PORT` honoured; the combined image now exits if either process dies instead of reporting healthy with half the app gone.

**Frontend** 🟠 → replaced outright
Streamlit is gone. The UI is now a **React 18 + TypeScript SPA** built with Vite, with no UI-library dependency — a design-token stylesheet and hand-built components (56 kB gzipped JS, 6 kB CSS). Five views: Research, Documents, Evaluate, Usage, History. Details in §5.

This also removed the duplicated report renderer, the shared `"default"` thread ID, the missing cancel button, and the hardcoded Render apology in one move — those were all Streamlit-era workarounds.

**G-23 · Tavily search never worked either** 🔴 → fixed
Same class of bug as G-22, found the first time a real Groq key ran the pipeline. `langchain_tavily.TavilySearch` has **no `api_key` field** — it reads `TAVILY_API_KEY` from the process environment. pydantic-settings loads `.env` into the `Settings` object but never exports it, so every Tavily call failed pydantic validation, returned an error string, and produced a report with **zero sources**. With `SEARCH_PROVIDER=auto` and a key present, this was the *default* path.
Fixed by publishing the variable in `_tavily_ready()` and making `auto` verify the provider is genuinely usable rather than trusting that a key string exists. Verified: 8 real sources on the next run.

**G-24 · A malformed provider tool call destroyed the whole research pass** 🔴 → fixed
Groq intermittently emits a bad function call, rejected with `tool_use_failed`. The researcher's blanket `except` treated it as fatal, discarding every source already retrieved and degrading the entire report. Now caught specifically: the agent is asked to write up what it has, with tool calling disabled. Observed firing in a live run — it saved a retry pass that had already gathered 8 sources.

**G-9 · Dependency pinning** 🟠 → fixed
[backend/requirements.lock](backend/requirements.lock). G-22 and G-23 were both unpinned-contract failures, both silent, both destroying citation integrity. The image now builds from the lockfile.

**G-6 / G-7 · State did not survive a restart** 🔴 → fixed
The vector index, document catalog and conversation history now live under `DATA_DIR`, backed by an Azure Files share mounted into the container. The FAISS index is saved on every write and restored at startup; the catalog is a JSON file beside it.
*Verified live:* uploaded a document, restarted the container (destroying all in-process state), and both `/agent/documents` and semantic search still returned it.

**G-8 · RAG shipped disabled** 🟠 → fixed properly
Rather than accepting a 2.5 GB image, embeddings moved from `sentence-transformers` (torch) to **fastembed** (ONNX). Same model class, ~50 MB of runtime; final image is **1.07 GB** and the model is baked in so cold starts don't download it. `rag_available` is now `true` in production.

**G-17 · Logs weren't structured** 🟢 → fixed
`LOG_FORMAT=json` emits one JSON object per line with `severity` and any `extra` fields, which is what Log Analytics can actually index. Text remains the default for local work.

**Frontend · the UI looked frozen** 🟠 → fixed
The backend now emits `agent_start`, `tool_call` and `tool_result` while a node is still running, through a thread-local sink drained by the same queue as graph events. The UI renders a live activity feed showing each search query as it is issued.
*Verified live through nginx:* events at 0s, 1s, 2s, 5s, 8s, 10s… across a 73-second run, instead of three updates at the end.

**Evaluation harness** 🟢 → added
[backend/eval/run_eval.py](backend/eval/run_eval.py) runs a fixed golden set through the pipeline, scores each report with the existing judge, and exits non-zero if the mean drops below threshold **or any report cites nothing**. Deliberately outside the unit suite — it makes real LLM calls and costs money.

### 3.2 Still open — deliberately deferred

**LangGraph checkpointer is still in-process** 🟠
`MemorySaver` ([backend/core/memory.py](backend/core/memory.py)) holds graph checkpoints in memory. Conversation *history* is now persisted (SQLite on the mounted share) and so is the vector index, but LangGraph's own checkpoint state is not — mid-graph resumption won't survive a restart. Nothing currently depends on that; it starts to matter if you add human-in-the-loop interrupts.

**Rate limiter is per-instance** 🟢
An in-process sliding window. Correct at `max-replicas 1`, which is the deployed configuration. Move it to Redis before scaling out, or the effective limit multiplies by replica count.

**SQLite over SMB** 🟢
Azure Files is an SMB share, and SQLite's locking over SMB is not ideal for concurrent writers. Fine at one replica; revisit together with the rate limiter if scaling out.

**G-16 · `chunk_text` offsets are O(n²) and can be wrong** 🟢 — [backend/core/rag.py:73-79](backend/core/rag.py#L73-L79). Cosmetic; `char_start` isn't consumed anywhere yet.

**G-18 · Test coverage** 🟠 — 21 → **51 tests**. The bugs that mattered now have direct coverage: gap extraction and source collection in [test_provenance.py](backend/tests/test_provenance.py), the event sink and JSON formatter in [test_events.py](backend/tests/test_events.py), catalog persistence and the RAG capability gate in [test_vectorstore.py](backend/tests/test_vectorstore.py). Still nothing covering graph-level routing end to end.

**G-21 · CI** 🟠 — now three jobs: backend lint+test, frontend typecheck+build, and **Docker image builds** for both services. Still no coverage gate and no `pip-audit`. The golden-set eval exists but isn't scheduled, because each run spends real money.

---

## 4. Backend plan (remaining)

Phase B1 (truthfulness) and most of B3 (hardening) landed in this pass. What's left:

1. **Pin dependencies** — `pip-compile` both requirement sets. Prevents the next G-22.
2. **Storage interface** — `core/storage.py` with `MemoryStore` / `DocumentStore` / `VectorStore` protocols, SQLite+FAISS locally and Postgres+pgvector when there's budget. Do this behind an interface *now* so the migration is a config change later.
3. **Structured JSON logging** with severity and `thread_id` fields.
4. **Graph-level tests** — retry fires once and only once; error path short-circuits to writer; gap list actually reaches the researcher.
5. **Golden-set eval** — ~20 fixed queries scored nightly by the existing judge, tracking drift. The judge is already built and now has a UI; wiring it to CI is cheap.
6. **`/ready` vs `/health`** — liveness versus dependency check.

---

## 5. Frontend — React SPA

Streamlit has been removed. The UI is a React 18 + TypeScript SPA served by nginx.

### Why the rewrite happened after all

The earlier advice here was "don't rewrite yet." That was right for a UI whose only job was rendering a report. It stopped being right once the app needed a credential: **Streamlit kept the API key server-side, and a naive SPA cannot.** Solving that properly required an nginx layer in front — and once there is a static-hosting layer, Streamlit's server is pure overhead.

The rewrite bought three things Streamlit could not:

1. **The key never reaches the browser.** nginx proxies `/api/*` and attaches `X-API-Key` itself. Same-origin, so CORS disappears entirely too.
2. **Real interaction control** — a working cancel button via `AbortController`, which Streamlit's rerun model makes awkward.
3. **A pipeline view that animates**, rather than three lines of text appearing over a minute.

### Stack and structure

| Choice | Rationale |
|---|---|
| Vite + React 18 + TS | Standard, fast builds, strict typing across the API boundary |
| **No UI library** | The whole surface is ~10 components; a design-token stylesheet is smaller and fully controllable. Total bundle 56 kB gzipped |
| Plain CSS with tokens | One `tokens.css` drives light and dark; components never hold raw colour values |
| `fetch` + `ReadableStream` for SSE | `EventSource` cannot POST, and the stream endpoint is a POST |

```
src/
  api/client.ts     typed client, SSE record parser, abort support
  types.ts          mirrors the Pydantic models
  components/       ui primitives · Pipeline · ReportView
  views/            Research · Documents · Evaluate · Usage · History
  hooks/            persisted settings + theme (light/dark/system)
  styles/           tokens.css · global.css
```

### Engineering notes worth keeping

- **SSE buffering.** The stream is split on the `\n\n` record separator, not per network chunk — a chunk boundary can land mid-JSON, and parsing per-chunk drops events under load.
- **Theme flash.** An inline script in `index.html` applies the stored theme before first paint; doing it in React means a visible flash of the wrong palette.
- **`add_header` inheritance.** nginx drops *all* inherited `add_header` directives in any block that declares one. The security headers are therefore repeated in each `location` that sets `Cache-Control`. This was caught by testing the live response, not by reading the config.
- **Cache strategy.** Hashed assets are `immutable` for a year; `index.html` is `no-store`, otherwise clients keep booting an old bundle against a new API.
- **Accessibility.** Keyboard-only focus rings, `aria-live` on pipeline status, skip link, `prefers-reduced-motion` honoured, meters carry proper roles.

### Still open

1. **Token-level streaming** — the backend emits one event per *agent*, so the pipeline shows three updates across ~60 s. `astream_events` would give per-token output. This is a backend change, and the UI is already structured for it.
2. **Live tool calls** — "🔍 Searching: *multi-agent benchmarks 2026*". `StreamEvent` declares a `tool_call` type that is still never emitted. The most convincing thing an agent UI can show.
3. **Report library** — persisted, searchable past reports. Needs G-6 first.

---

## 6. Deployment — live on Azure Container Apps

### Why Azure, and why Container Apps

The subscription is **Azure for Students** — $100 credit, no credit card, no expiry cliff for 12 months. That changes the calculus that ruled out GCP: there, the piece worth paying for (Cloud SQL) had no free tier and needed a card.

Container Apps was the right service:

| Option | Verdict |
|---|---|
| **Container Apps (chosen)** | Free grant of 180k vCPU-s / 360k GiB-s / 2M requests per month. Scale-to-zero, so idle costs nothing. Native SSE + websockets. |
| App Service F1 | 60 CPU-min/**day** and 1 GB RAM. The pipeline would exhaust that in a handful of reports. |
| AKS | Wrong altitude — a cluster to operate for two containers. |

### What is actually running

| Resource | Name | Notes |
|---|---|---|
| Resource group | `research-agent-rg` | `centralindia` |
| Registry | `racr1f72eb6f326c` | ACR Basic, **admin user disabled** |
| Environment | `research-agent-env` | Consumption workload profile |
| API | `research-agent-api` | 1 vCPU / 2 GiB, min 0 / max 1, port 8000 |
| UI | `research-agent-ui` | 0.5 vCPU / 1 GiB, min 0 / max 1, port 8080 (nginx + React) |

- **API** — `https://research-agent-api.agreeablestone-a7a39990.centralindia.azurecontainerapps.io`
- **UI** — `https://research-agent-ui.agreeablestone-a7a39990.centralindia.azurecontainerapps.io`

Reproduce or redeploy with [deploy/azure.sh](deploy/azure.sh).

**`max-replicas=1` is deliberate.** The vector index, the LangGraph checkpointer and the SQLite history are all in process memory (G-6/G-7). A second replica would serve requests that cannot see the first replica's uploads. Raise this only after that state moves out.

**Registry access uses each app's system-assigned managed identity** with an `AcrPull` role assignment; ACR's admin user is disabled, so there is no registry password stored in the app config or recoverable from it.

**Cost:** Container Apps compute sits inside the free monthly grant at this volume, and both apps scale to zero. ACR Basic (~$5/month) is the only standing charge — swap it for ghcr.io if you want a strict $0 footprint. Everything else is within the free grant.

### Three things the deployment surfaced that local testing had not

**1. `az acr build` is blocked on student subscriptions.** ACR Tasks returns `TasksOperationsNotAllowed`. The deploy script therefore builds locally and pushes; a Docker daemon is required.

**2. `/health` was advertising a capability the image did not have.** It reported `rag_available: true` on an image with no faiss and no embedding stack, because the import guard in `main.py` only proved `core.vectorstore` imports — and that module defers its heavy imports to call time. Uploads returned `200` and indexed nothing; search then returned `404`. Fixed with `vectorstore.dependencies_available()`, which probes the real dependencies. The endpoint now honestly returns `501`.

**3. Setting `ALLOWED_ORIGINS` crashed the container on startup — a bug I introduced in this same pass.** `pydantic-settings` JSON-decodes complex field types straight from the environment source *before* any `field_validator` runs, so typing the field as `List[str]` made `ALLOWED_ORIGINS=https://…` raise `SettingsError` and exit 1 in about a second. It is now a plain `str` with an `allowed_origins_list` property, covered by [backend/tests/test_config.py](backend/tests/test_config.py).

That third one is worth dwelling on: it would have broken **every** deployment that followed this guide's own instruction to lock CORS down before going public, and it was invisible to the 35 local tests because none of them set that variable.

### And two more from the React deployment

**4. nginx does not send SNI to HTTPS upstreams by default.** Azure Container Apps' ingress routes on SNI and closes the connection mid-handshake without it, which surfaces only as a generic `502 Bad Gateway`. The error log said `peer closed connection in SSL handshake while SSL handshaking to upstream`. Fixed with `proxy_ssl_server_name on` + `proxy_ssl_name`.

**5. `wait -n` is not available in `dash`.** The all-in-one Dockerfile used `sh -c "... wait -n; kill 0"`, and `python:3.11-slim`'s `/bin/sh` is dash — the container died instantly with `wait: Illegal option -n`. Switched to `bash -c`. This one had been sitting in the root Dockerfile unnoticed because that image is a local convenience path nobody had booted.

### Operating it

```bash
# Rotate in a working Groq key (the current one returns 401)
az containerapp secret set -g research-agent-rg -n research-agent-api \
  --secrets groq-key=<YOUR_KEY>
az containerapp revision restart -g research-agent-rg -n research-agent-api \
  --revision $(az containerapp show -g research-agent-rg -n research-agent-api \
               --query properties.latestRevisionName -o tsv)

# Logs
az containerapp logs show -g research-agent-rg -n research-agent-api --type console --tail 50
az containerapp logs show -g research-agent-rg -n research-agent-api --type system  --tail 50

# Tear everything down
az group delete -n research-agent-rg --yes
```

The API key is in `%TEMP%\ra_apikey.txt` on this machine and is set as the `api-key` secret on both apps — the API validates it, the UI's nginx injects it. Browser users never need it and never receive it; it only matters for calling the API directly.

Rotating it takes a set on each app plus a restart, since running containers cache secret values:

```bash
NEW=$(openssl rand -hex 24)
for APP in research-agent-api research-agent-ui; do
  az containerapp secret set -g research-agent-rg -n $APP --secrets api-key=$NEW
  az containerapp revision restart -g research-agent-rg -n $APP \
    --revision $(az containerapp show -g research-agent-rg -n $APP \
                 --query properties.latestRevisionName -o tsv)
done
```

### Enabling RAG later

The slim image deliberately omits faiss/sentence-transformers (~2 GB with torch, and a scale-to-zero app pays that on every cold start). To turn RAG on: add `RUN pip install -r requirements-rag.txt` to [backend/Dockerfile](backend/Dockerfile), bake the embedding model into the image so cold start does not download 90 MB, and keep the app at 2 GiB. `/health` will then report `rag_available: true` truthfully.

---

## 7. Sequencing from here

| Priority | Work | Effort |
|---|---|---|
| **Now** | Rotate in a valid Groq key — the deployed API returns 401 on every LLM call | 5 min |
| **Next** | Pin dependencies (G-9) | 1 hr |
| **Next** | Cosmos DB free tier for G-6/G-7 (memory + checkpoints + vectors) | 1–2 days |
| **Then** | Token-level streaming + live tool calls (F2) | 1 day |
| **Then** | Graph tests + golden-set eval in CI | 1 day |
| **Later** | The document-fusion wedge (§1) | ~5 days |

Authentication and CORS are now enforced on the deployed instance, and the `ddgs` fix is live, so the two urgent liabilities from the first review are closed. The single remaining blocker to a *working* deployment is the Groq key.

---

## 8. Verified during this pass

Run against the real code, not asserted from reading:

- `ruff check backend/ frontend/ --ignore E501` → **All checks passed**
- `pytest tests/ -q` → **38 passed** (was 21; +17 covering provenance, gap extraction, measured tokens, settings parsing)
- **Live on Azure (API)** → `/health` 200 with `rag_available:false` + `auth_required:true`; `/agent/graph` 401 without key and 200 with it; `/agent/upload` honest 501; rate limiter allowed exactly 20 then returned 429
- **Live on Azure (UI)** → SPA shell 200 with correct `<title>`; hashed assets 200 with `immutable`; SPA fallback 200 on a deep client route; `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy` all present
- **Proxy** → `/api/health`, `/api/agent/graph`, `/api/agent/usage` all 200 **with no key sent by the client**, while the API still returns 401 to an unauthenticated direct call — proving injection happens server-side
- **SSE through nginx** → `Content-Type: text/event-stream`, `Transfer-Encoding: chunked`, records arriving individually rather than as one buffered blob
- **Frontend build** → `tsc -b` clean under `strict` + `noUncheckedIndexedAccess`; 174 kB JS (56 kB gzip), 27 kB CSS (6 kB gzip)
- **All-in-one image** → boots, serves the SPA and proxies `/api/health` (after fixing the `dash`/`wait -n` bug)
- **Both container apps** report `RunningAtMaxScale` / `Healthy` on managed-identity registry pull with ACR admin disabled
- **Live search** → `web_search` returned 4 real URLs and collected all 4 with provider attribution
- **Old code path** → confirmed `ImportError: Could not import ddgs`, proving search was broken before this pass
- **Full graph E2E** (real LangGraph + real DuckDuckGo + stubbed LLM) → 7 real URLs accumulated across two research passes; model-invented citation excluded; retry prompt carried the literal gap text; per-agent token breakdown correct
- **Middleware smoke test** → 401 on missing/bad key, 200 on good key, 429 past the rate limit, traversal filename rejected, JSON search body accepted, streaming saved history and recorded usage
- `python -m py_compile frontend/app.py` → clean

**Not verified:** the live Groq LLM path. The key in `backend/.env` returns `401 Invalid API Key`, both locally and on Azure, so every end-to-end run used a stubbed model. Token capture is proven against a synthetic `usage_metadata` payload shaped like `ChatGroq`'s, not against a real Groq response — worth one live run once the key is refreshed. The deployed pipeline correctly degrades rather than crashing: it returns a "Report Generation Failed" report with `needs_human_review: true`, which is the guardrail behaving as designed.

**No commits, no pushes, no branches.** All source changes are working-tree only. The Azure resources listed in §6 are real and were created during this session.
