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


class FinalReport(BaseModel):
    title: str
    summary: str
    research_findings: List[str]
    analysis: List[str]
    conclusion: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=1000)
    thread_id: str = Field(default="default")
    stream: bool = False


class ChatResponse(BaseModel):
    thread_id: str
    report: FinalReport
    latency_ms: float
    iterations: int