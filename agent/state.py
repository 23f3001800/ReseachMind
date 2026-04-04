# agent/state.py

from typing import TypedDict, List, Optional
from langgraph.graph import MessagesState

class ResearchState(TypedDict):
    # Input
    topic: str
    depth: str                    # "quick" | "deep"

    # Planning
    search_queries: List[str]     # 3 queries the planner generates

    # Research
    search_results: List[dict]    # raw results from web search
    extracted_facts: List[str]    # key facts pulled from results
    rag_context: str              # relevant chunks from knowledge base

    # Output
    report_sections: dict         # {intro, findings, analysis, conclusion}
    final_report: str             # formatted markdown report
    citations: List[str]          # source URLs

    # Control
    error: Optional[str]
    steps_taken: List[str]        # audit trail of what happened