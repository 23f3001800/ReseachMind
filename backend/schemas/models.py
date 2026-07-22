from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"


class GuardedOutput(BaseModel):
    """Every agent output must pass through this schema."""
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    review_reason: Optional[str] = None
    sources: List[str] = []
    agent: AgentRole


class ResearchResult(BaseModel):
    topic: str
    findings: List[str]
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    topic: str
    key_insights: List[str]
    data_points: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class Source(BaseModel):
    """A page the system actually retrieved during research.

    Only ever built from executed search results — never from model output —
    so every citation in a report is traceable.
    """
    url: str
    title: str = ""
    provider: str = ""


class FinalReport(BaseModel):
    title: str
    summary: str
    research_findings: List[str]
    analysis: List[str]
    conclusion: str
    sources: List[Source] = []
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False


class WriterReport(BaseModel):
    """Schema for LLM structured output — only contains fields the LLM produces."""
    title: str = Field(description="A concise, descriptive report title")
    summary: str = Field(description="2-3 sentence executive summary")
    research_findings: List[str] = Field(description="Key factual findings from the research")
    analysis: List[str] = Field(description="Analytical insights and patterns")
    conclusion: str = Field(description="Final conclusion paragraph")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=1000)
    thread_id: str = Field(default="default")
    stream: bool = False


class ChatResponse(BaseModel):
    thread_id: str
    report: FinalReport
    latency_ms: float
    iterations: int
    token_usage: Optional[dict] = None


class SearchRequest(BaseModel):
    """Body for semantic search over uploaded documents."""
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class StreamEvent(BaseModel):
    """Server-Sent Event payload for streaming agent progress."""
    event: Literal[
        "agent_start", "agent_end", "tool_call", "tool_result", "error", "complete"
    ]
    agent: Optional[str] = None
    content: Optional[str] = None
    data: Optional[dict] = None