from typing import Dict, List, Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state across all agents in the graph."""
    messages:          Annotated[List[BaseMessage], add_messages]
    query:             str
    research_output:   Optional[str]
    analysis_output:   Optional[str]
    final_report:      Optional[Dict]
    sources:           List[Dict]      # {url, title, provider} — retrieved, not generated
    confidence:        float
    needs_human_review: bool
    review_reason:     Optional[str]
    iterations:        int
    next_agent:        Optional[str]
    research_gaps:     bool            # True if analyst found significant gaps
    research_gaps_detail: List[str]    # The gaps themselves, fed to the retry pass
    retry_count:       int             # Number of researcher retries so far
    token_usage:       Optional[Dict]  # Measured token/cost totals for this run