import time
import os
import asyncio
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from schemas.models import ChatRequest, ChatResponse, FinalReport, StreamEvent
from core.supervisor import run_agent, run_agent_stream
from core.memory import get_conversation_history, save_to_history, clear_thread
from core.logger import get_logger
from core.rag import load_document, load_text, chunk_text
from core import vectorstore
from config import settings

logger = get_logger(__name__)

# Maximum time (seconds) to wait for the agent pipeline
REQUEST_TIMEOUT = 120


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
    logger.info("Agentic Research Assistant starting...")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Agentic Research Assistant",
    description="Multi-agent research system: Researcher → Analyst → Writer with memory and guardrails",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a structured error response."""
    logger.error(f"Unhandled error | path={request.url.path} error={exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentic-research-assistant"}


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
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f"Pipeline timed out after {REQUEST_TIMEOUT}s | thread_id={request.thread_id}")
        raise HTTPException(
            status_code=504,
            detail=f"Agent pipeline timed out after {REQUEST_TIMEOUT} seconds.",
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
        f"needs_review={report.needs_human_review}"
    )

    return ChatResponse(
        thread_id=request.thread_id,
        report=report,
        latency_ms=latency_ms,
        iterations=result.get("iterations", 0),
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
        import json
        start = time.perf_counter()
        last_state = {}

        try:
            async for node_name, node_output in run_agent_stream(
                query=request.message,
                thread_id=request.thread_id,
            ):
                last_state.update(node_output)

                # Emit agent completion event
                event = StreamEvent(
                    event="agent_end",
                    agent=node_name,
                    content=f"{node_name} completed",
                    data={
                        "confidence": node_output.get("confidence"),
                        "iterations": node_output.get("iterations"),
                    },
                )
                yield f"data: {event.model_dump_json()}\n\n"

            # Final complete event with full report
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            report_data = last_state.get("final_report")

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
            "request timeout (120s)",
            "self-reflection retry loop",
        ],
    }


# ── RAG: Document Upload ──────────────────────────────────
# In-memory document store (per session — replaced by vector DB in commit 17)
_document_store: dict = {}


@app.post("/agent/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, TXT, MD) for RAG-enhanced research."""
    allowed_exts = {".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_exts}",
        )

    # Save to temp location
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = load_document(file_path)
        chunks = chunk_text(text)

        # Store chunks in memory (keyed by filename)
        _document_store[file.filename] = {
            "text_length": len(text),
            "chunks": chunks,
        }

        # Index in vector store for semantic search
        try:
            vectorstore.add_documents(chunks, source=file.filename)
        except ImportError as ie:
            logger.warning(f"Vector indexing skipped — {ie}")

        logger.info(
            f"Document uploaded | file={file.filename} "
            f"chars={len(text)} chunks={len(chunks)}"
        )

        return {
            "filename": file.filename,
            "text_length": len(text),
            "num_chunks": len(chunks),
            "chunk_preview": chunks[0]["content"][:200] if chunks else "",
        }
    except Exception as e:
        logger.error(f"Upload failed | file={file.filename} error={e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/agent/documents")
async def list_documents():
    """List uploaded documents and their chunk counts."""
    return {
        "documents": [
            {
                "filename": name,
                "text_length": data["text_length"],
                "num_chunks": len(data["chunks"]),
            }
            for name, data in _document_store.items()
        ],
        "count": len(_document_store),
    }


@app.post("/agent/search")
async def search_documents(query: str, k: int = 5):
    """Semantic search across uploaded documents."""
    if not vectorstore.has_documents():
        raise HTTPException(status_code=404, detail="No documents indexed. Upload a file first.")

    try:
        results = vectorstore.search(query, k=k)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search failed | error={e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")