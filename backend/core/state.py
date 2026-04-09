from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state across all agents in the graph."""
    messages:          Annotated[List[BaseMessage], add_messages]
    query:             str
    research_output:   Optional[str]
    analysis_output:   Optional[str]
    final_report:      Optional[str]
    sources:           List[str]
    confidence:        float
    needs_human_review: bool
    review_reason:     Optional[str]
    iterations:        int
    next_agent:        Optional[str]