from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.state import AgentState
from core.memory import get_checkpointer
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from config import settings


def route_after_supervisor(state: AgentState) -> str:
    """Supervisor decides next node based on state."""
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "END":
        return END
    return next_agent


def route_from_researcher(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "writer"
    return state.get("next_agent", "analyst")


def route_from_analyst(state: AgentState) -> str:
    return state.get("next_agent", "writer")


def build_graph():
    """Build and compile the multi-agent LangGraph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    # Entry point
    graph.set_entry_point("researcher")

    # Conditional routing
    graph.add_conditional_edges(
        "researcher",
        route_from_researcher,
        {"analyst": "analyst", "writer": "writer"},
    )
    graph.add_conditional_edges(
        "analyst",
        route_from_analyst,
        {"writer": "writer"},
    )
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=get_checkpointer())


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, thread_id: str = "default") -> AgentState:
    """Run the full multi-agent pipeline for a query."""
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "research_output": None,
        "analysis_output": None,
        "final_report": None,
        "sources": [],
        "confidence": 1.0,
        "needs_human_review": False,
        "review_reason": None,
        "iterations": 0,
        "next_agent": "researcher",
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)
    return result