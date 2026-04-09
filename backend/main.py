import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from schemas.models import ChatRequest, ChatResponse, FinalReport
from core.supervisor import run_agent
from core.memory import get_conversation_history, save_to_history, clear_thread
import re


def parse_report(raw: str, sources: list, confidence: float, needs_review: bool) -> FinalReport:
    """Parse raw writer output into FinalReport schema."""
    def extract(tag: str) -> str:
        pattern = rf"{tag}:\s*(.*?)(?=\n[A-Z ]+:|$)"
        match = re.search(pattern, raw, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_list(tag: str):
        section = extract(tag)
        return [
            re.sub(r"^\d+\.\s*|-\s*", "", line).strip()
            for line in section.split("\n")
            if line.strip() and len(line.strip()) > 3
        ]

    return FinalReport(
        title=extract("TITLE") or "Research Report",
        summary=extract("SUMMARY") or raw[:300],
        research_findings=extract_list("KEY FINDINGS"),
        analysis=extract_list("ANALYSIS"),
        conclusion=extract("CONCLUSION") or "",
        sources=sources,
        confidence=confidence,
        needs_human_review=needs_review,
    )


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
        result = run_agent(
            query=request.message,
            thread_id=request.thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    raw_report = result.get("final_report", "")
    if not raw_report:
        raise HTTPException(status_code=500, detail="Agent produced no output.")

    report = parse_report(
        raw=raw_report,
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.5),
        needs_review=result.get("needs_human_review", False),
    )

    # Save to memory
    save_to_history(request.thread_id, request.message, raw_report)

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