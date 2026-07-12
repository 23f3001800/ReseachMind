import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from schemas.models import ChatRequest, ChatResponse, FinalReport
from core.supervisor import run_agent
from core.memory import get_conversation_history, save_to_history, clear_thread


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Agentic Research Assistant starting...")
    yield
    print("Shutting down.")


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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentic-research-assistant"}


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run the full Researcher → Analyst → Writer pipeline."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start = time.perf_counter()

    try:
        result = await run_agent(
            query=request.message,
            thread_id=request.thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    report_data = result.get("final_report")
    if not report_data:
        raise HTTPException(status_code=500, detail="Agent produced no output.")

    # Build FinalReport from structured output dict + pipeline metadata
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

    # Save to memory
    save_to_history(request.thread_id, request.message, report.summary)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return ChatResponse(
        thread_id=request.thread_id,
        report=report,
        latency_ms=latency_ms,
        iterations=result.get("iterations", 0),
    )


@app.get("/agent/history/{thread_id}")
async def get_history(thread_id: str):
    """Retrieve conversation history for a thread."""
    history = get_conversation_history(thread_id)
    return {"thread_id": thread_id, "exchanges": history, "count": len(history)}


@app.delete("/agent/history/{thread_id}")
async def clear_history(thread_id: str):
    """Clear memory for a thread."""
    clear_thread(thread_id)
    return {"message": f"Thread {thread_id} cleared."}


@app.get("/agent/graph")
async def get_graph_info():
    """Show agent graph structure."""
    return {
        "agents": ["researcher", "analyst", "writer"],
        "flow": "researcher → analyst → writer",
        "routing": "conditional — low confidence skips to writer",
        "memory": "per thread_id via MemorySaver checkpointer",
        "guardrails": [
            "confidence threshold check",
            "agent error fallback",
            "human review flag",
        ],
    }