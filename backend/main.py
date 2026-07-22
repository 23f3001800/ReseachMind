import time
import os
import asyncio
import threading
import uuid
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from schemas.models import ChatRequest, ChatResponse, FinalReport, StreamEvent, SearchRequest
from core.supervisor import run_agent, run_agent_stream, FINAL_META_NODE, PROGRESS_NODE
from core.memory import get_conversation_history, save_to_history, clear_thread
from core.logger import get_logger
from core.usage import usage_tracker, RequestMetrics
from config import settings

# Optional RAG imports — gracefully degrade if ML deps not installed.
# The import succeeding is not enough: core.vectorstore defers faiss and the
# embedding stack to call time, so it imports fine on a slim image where
# indexing would fail. Ask the module whether it can actually do the work.
try:
    from core.rag import load_document, chunk_text
    from core import vectorstore
    RAG_AVAILABLE = vectorstore.dependencies_available()
except ImportError:
    RAG_AVAILABLE = False


logger = get_logger(__name__)

# Endpoints that stay open even when an API key is configured.
_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def _setup_langsmith():
    """Configure LangSmith tracing if API key is available."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(f"LangSmith tracing enabled | project={settings.langchain_project}")
    elif settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        logger.info("LangSmith tracing flag set but no API key provided")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_langsmith()
    if not settings.api_key:
        logger.warning(
            "API_KEY is not set — this instance is open to anyone who finds its URL. "
            "Set API_KEY before exposing it publicly."
        )
    if RAG_AVAILABLE:
        # Restore a previously persisted index so uploads survive a restart.
        # Off the request path: the first search would otherwise pay for it.
        await asyncio.to_thread(vectorstore.initialize)
    logger.info(f"researchMind starting | rag={RAG_AVAILABLE} auth={bool(settings.api_key)}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="researchMind",
    description="Multi-agent research system: Researcher → Analyst → Writer with memory, guardrails and traceable citations",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API key + rate limiting ──────────────────────────────
# In-process limiter. Correct for a single instance, which is what the free
# tiers give you; if this ever runs multi-instance, move it to Redis.
_rate_lock = threading.Lock()
_request_log: dict = defaultdict(deque)


def _rate_limited(client_id: str) -> bool:
    """Sliding 60s window per client. True if the caller is over the limit."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return False

    now = time.monotonic()
    with _rate_lock:
        hits = _request_log[client_id]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            return True
        hits.append(now)

        # Keep the dict from growing without bound across many client IPs
        if len(_request_log) > 10_000:
            for key in [k for k, v in _request_log.items() if not v]:
                del _request_log[key]
    return False


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    """Reject unauthenticated or over-quota callers before any LLM spend."""
    if request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    if settings.api_key:
        provided = request.headers.get("X-API-Key", "")
        if provided != settings.api_key:
            logger.warning(f"Rejected request | path={request.url.path} reason=bad_api_key")
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-API-Key."})

    client_id = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
    if _rate_limited(client_id):
        logger.warning(f"Rate limited | path={request.url.path}")
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded ({settings.rate_limit_per_minute}/min). Try again shortly."},
        )

    return await call_next(request)


# ── Global exception handler ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a structured error response."""
    logger.error(f"Unhandled error | path={request.url.path} error={exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


def _describe_progress(evt: dict) -> str:
    """Human-readable one-liner for an in-flight progress event."""
    kind = evt.get("event", "")
    data = evt.get("data") or {}
    agent = (evt.get("agent") or "agent").title()

    if kind == "tool_call":
        return f"Searching: {data.get('query', '')}"
    if kind == "tool_result":
        found = data.get("new_sources", 0)
        return f"Found {found} new source{'' if found == 1 else 's'}"
    if kind == "agent_start":
        pass_no = data.get("pass_number")
        return f"{agent} started" + (f" (pass {pass_no})" if pass_no and pass_no > 1 else "")
    return kind


def _record_usage(thread_id: str, query: str, token_usage: dict, latency_ms: float):
    """Record measured token usage from a completed pipeline run."""
    if not token_usage:
        return
    metrics = RequestMetrics(
        thread_id=thread_id,
        query=query[:100],
        latency_ms=latency_ms,
        total_input_tokens=token_usage.get("total_input_tokens", 0),
        total_output_tokens=token_usage.get("total_output_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
        estimated_cost_usd=token_usage.get("estimated_cost_usd", 0.0),
        agent_metrics=token_usage.get("agent_breakdown", {}),
    )
    usage_tracker.record(metrics)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "researchmind",
        "rag_available": RAG_AVAILABLE,
        "auth_required": bool(settings.api_key),
    }


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run the full Researcher → Analyst → Writer pipeline."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"Chat request | thread_id={request.thread_id} query='{request.message[:80]}'")
    start = time.perf_counter()

    # Run with timeout to prevent indefinite hangs
    try:
        result = await asyncio.wait_for(
            run_agent(
                query=request.message,
                thread_id=request.thread_id,
            ),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"Pipeline timed out after {settings.request_timeout}s | thread_id={request.thread_id}")
        raise HTTPException(
            status_code=504,
            detail=f"Agent pipeline timed out after {settings.request_timeout} seconds.",
        )
    except Exception as e:
        logger.error(f"Pipeline failed | thread_id={request.thread_id} error={e}")
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    report_data = result.get("final_report")
    if not report_data:
        raise HTTPException(status_code=500, detail="Agent produced no output.")

    # Build FinalReport — handle malformed data defensively
    try:
        report = FinalReport(
            title=report_data.get("title", "Research Report"),
            summary=report_data.get("summary", ""),
            research_findings=report_data.get("research_findings", []),
            analysis=report_data.get("analysis", []),
            conclusion=report_data.get("conclusion", ""),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.5),
            needs_human_review=result.get("needs_human_review", False),
        )
    except Exception as e:
        logger.error(f"Failed to parse report | error={e} data={report_data}")
        raise HTTPException(status_code=500, detail="Failed to parse agent output into report.")

    # Save to memory (non-blocking — don't fail the request if this errors)
    try:
        save_to_history(request.thread_id, request.message, report.summary)
    except Exception as e:
        logger.warning(f"Failed to save history | thread_id={request.thread_id} error={e}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        f"Chat response | thread_id={request.thread_id} "
        f"confidence={report.confidence} latency_ms={latency_ms} "
        f"sources={len(report.sources)} needs_review={report.needs_human_review}"
    )

    token_usage = result.get("token_usage") or {}
    _record_usage(request.thread_id, request.message, token_usage, latency_ms)

    return ChatResponse(
        thread_id=request.thread_id,
        report=report,
        latency_ms=latency_ms,
        iterations=result.get("iterations", 0),
        token_usage=token_usage,
    )


@app.get("/agent/history/{thread_id}")
async def get_history(thread_id: str):
    """Retrieve conversation history for a thread."""
    try:
        history = get_conversation_history(thread_id)
    except Exception as e:
        logger.error(f"Failed to fetch history | thread_id={thread_id} error={e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")
    return {"thread_id": thread_id, "exchanges": history, "count": len(history)}


@app.delete("/agent/history/{thread_id}")
async def clear_history(thread_id: str):
    """Clear memory for a thread."""
    try:
        clear_thread(thread_id)
    except Exception as e:
        logger.error(f"Failed to clear history | thread_id={thread_id} error={e}")
        raise HTTPException(status_code=500, detail="Failed to clear history.")
    return {"message": f"Thread {thread_id} cleared."}


@app.post("/agent/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream agent pipeline progress as Server-Sent Events."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"Stream request | thread_id={request.thread_id} query='{request.message[:80]}'")

    async def event_generator():
        start = time.perf_counter()
        last_state = {}

        try:
            async for node_name, node_output in run_agent_stream(
                query=request.message,
                thread_id=request.thread_id,
            ):
                # In-flight progress from inside a node (agent_start, tool_call,
                # tool_result). Forwarded as-is and NOT merged into last_state —
                # it is telemetry about the run, not graph state.
                if node_name == PROGRESS_NODE:
                    progress = StreamEvent(
                        event=node_output.get("event", "tool_call"),
                        agent=node_output.get("agent") or None,
                        content=_describe_progress(node_output),
                        data=node_output.get("data") or {},
                    )
                    yield f"data: {progress.model_dump_json()}\n\n"
                    continue

                last_state.update(node_output)

                # The trailing meta event carries telemetry, not agent progress
                if node_name == FINAL_META_NODE:
                    continue

                # Emit agent completion event
                event = StreamEvent(
                    event="agent_end",
                    agent=node_name,
                    content=f"{node_name} completed",
                    data={
                        "confidence": node_output.get("confidence"),
                        "iterations": node_output.get("iterations"),
                        "sources_found": len(node_output.get("sources") or []),
                    },
                )
                yield f"data: {event.model_dump_json()}\n\n"

            # Final complete event with full report
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            report_data = last_state.get("final_report")
            token_usage = last_state.get("token_usage") or {}

            # Streaming is the default path in the UI, so history and usage must
            # be recorded here too — not only in the non-streaming endpoint.
            if report_data:
                try:
                    save_to_history(
                        request.thread_id,
                        request.message,
                        report_data.get("summary", ""),
                    )
                except Exception as e:
                    logger.warning(f"Failed to save history | thread_id={request.thread_id} error={e}")

            _record_usage(request.thread_id, request.message, token_usage, elapsed)

            complete_event = StreamEvent(
                event="complete",
                content="Pipeline finished",
                data={
                    "report": report_data,
                    "sources": last_state.get("sources", []),
                    "confidence": last_state.get("confidence", 0.5),
                    "needs_human_review": last_state.get("needs_human_review", False),
                    "iterations": last_state.get("iterations", 0),
                    "latency_ms": elapsed,
                    "token_usage": token_usage,
                },
            )
            yield f"data: {complete_event.model_dump_json()}\n\n"

        except Exception as e:
            logger.error(f"Stream error | error={e}")
            error_event = StreamEvent(event="error", content=str(e))
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/graph")
async def get_graph_info():
    """Show agent graph structure."""
    return {
        "agents": ["researcher", "analyst", "writer"],
        "flow": "researcher → analyst → writer (with optional retry loop)",
        "routing": "conditional — low confidence skips to writer, gaps trigger re-research",
        "memory": "SQLite-backed conversation history + MemorySaver checkpointer",
        "guardrails": [
            "confidence threshold check",
            "agent error fallback",
            "human review flag",
            f"request timeout ({settings.request_timeout}s)",
            "self-reflection retry loop targeting identified gaps",
            "citations restricted to retrieved sources",
        ],
    }


# ── RAG: Document Upload ──────────────────────────────────
# The document catalog is persisted next to the vector index rather than held
# in a module global, so uploads survive a restart or scale-to-zero. Point
# DATA_DIR at a mounted volume for that to mean anything in production.
ALLOWED_UPLOAD_EXTS = {".pdf", ".txt", ".md"}


@app.post("/agent/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, TXT, MD) for RAG-enhanced research."""
    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="RAG not available — install optional deps: pip install -r requirements-rag.txt",
        )

    original_name = Path(file.filename or "").name  # strips any directory component
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext or 'none'}. Allowed: {sorted(ALLOWED_UPLOAD_EXTS)}",
        )

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Limit is {settings.max_upload_mb} MB.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Write under a generated name so a crafted filename can't escape the
    # upload directory or overwrite anything.
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}{ext}"
    file_path.write_bytes(content)

    try:
        text = load_document(str(file_path))
        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in this file.")

        # Index in the vector store, then record it in the persisted catalog.
        # Catalog is written only after indexing succeeds, so it can never
        # advertise a document that isn't actually searchable.
        vectorstore.add_documents(chunks, source=original_name)

        catalog = vectorstore.load_catalog()
        catalog[original_name] = {
            "text_length": len(text),
            "num_chunks": len(chunks),
        }
        vectorstore.save_catalog(catalog)

        logger.info(
            f"Document uploaded | file={original_name} "
            f"chars={len(text)} chunks={len(chunks)}"
        )

        return {
            "filename": original_name,
            "text_length": len(text),
            "num_chunks": len(chunks),
            "chunk_preview": chunks[0]["content"][:200] if chunks else "",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed | file={original_name} error={e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        file_path.unlink(missing_ok=True)  # chunks are indexed; the raw file isn't needed


@app.get("/agent/documents")
async def list_documents():
    """List uploaded documents and their chunk counts."""
    if not RAG_AVAILABLE:
        return {"documents": [], "count": 0}

    catalog = vectorstore.load_catalog()
    return {
        "documents": [
            {
                "filename": name,
                "text_length": data.get("text_length", 0),
                "num_chunks": data.get("num_chunks", 0),
            }
            for name, data in catalog.items()
        ],
        "count": len(catalog),
    }


@app.post("/agent/search")
async def search_documents(request: SearchRequest):
    """Semantic search across uploaded documents."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=501, detail="RAG not available on this deployment.")
    if not vectorstore.has_documents():
        raise HTTPException(status_code=404, detail="No documents indexed. Upload a file first.")

    try:
        results = vectorstore.search(request.query, k=request.k)
        return {"query": request.query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search failed | error={e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/agent/evaluate")
async def evaluate_report_endpoint(request: ChatRequest):
    """Run a query through the pipeline and evaluate the output with LLM-as-judge."""
    from core.evaluator import evaluate_report

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"Evaluate request | query='{request.message[:80]}'")
    start = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            run_agent(query=request.message, thread_id=f"eval-{request.thread_id}"),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out during evaluation.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    report_data = result.get("final_report")
    if not report_data:
        raise HTTPException(status_code=500, detail="No report generated to evaluate.")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    _record_usage(request.thread_id, request.message, result.get("token_usage") or {}, latency_ms)

    evaluation = await asyncio.to_thread(
        evaluate_report,
        query=request.message,
        report=report_data,
        sources=result.get("sources", []),
    )

    if evaluation is None:
        raise HTTPException(status_code=500, detail="Evaluation failed.")

    return {
        "query": request.message,
        "report": report_data,
        "sources": result.get("sources", []),
        "evaluation": evaluation.model_dump(),
    }


# ── Analytics ─────────────────────────────────────────────
@app.get("/agent/usage")
async def get_usage():
    """Get cumulative token usage and cost analytics."""
    return usage_tracker.get_summary()


@app.delete("/agent/usage")
async def reset_usage():
    """Reset usage counters."""
    usage_tracker.reset()
    return {"message": "Usage tracker reset."}
