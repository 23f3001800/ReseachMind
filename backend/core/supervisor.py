import asyncio
import time
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.state import AgentState
from core.memory import get_checkpointer
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from core.logger import get_logger

logger = get_logger(__name__)


def route_after_supervisor(state: AgentState) -> str:
    """Supervisor decides next node based on state."""
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "END":
        return END
    return next_agent


def route_from_researcher(state: AgentState) -> str:
    if state.get("needs_human_review"):
        logger.info("Routing: researcher → writer (skipping analyst — needs review)")
        return "writer"
    next_node = state.get("next_agent", "analyst")
    logger.info(f"Routing: researcher → {next_node}")
    return next_node


def route_from_analyst(state: AgentState) -> str:
    next_node = state.get("next_agent", "writer")
    logger.info(f"Routing: analyst → {next_node}")
    return next_node


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
        {"writer": "writer", "researcher": "researcher"},
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


def _invoke_sync(query: str, thread_id: str) -> AgentState:
    """Synchronous graph invocation — runs in a thread pool."""
    start = time.perf_counter()
    logger.info(f"Pipeline started | thread_id={thread_id} query='{query[:80]}'")

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
        "research_gaps": False,
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)

    elapsed = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        f"Pipeline completed | thread_id={thread_id} "
        f"iterations={result.get('iterations', 0)} "
        f"confidence={result.get('confidence', 0)} "
        f"duration_ms={elapsed}"
    )
    return result


async def run_agent(query: str, thread_id: str = "default") -> AgentState:
    """Run the full multi-agent pipeline without blocking the async event loop."""
    return await asyncio.to_thread(_invoke_sync, query, thread_id)


def _stream_sync(query: str, thread_id: str):
    """Synchronous generator — yields (node_name, state) tuples as each node completes."""
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
        "research_gaps": False,
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream(initial_state, config=config):
        # event is a dict like {"researcher": {state_updates}} or {"analyst": {...}}
        for node_name, node_output in event.items():
            yield node_name, node_output


async def run_agent_stream(query: str, thread_id: str = "default"):
    """Async generator — yields (node_name, state) tuples without blocking the event loop."""
    import queue
    import threading

    q = queue.Queue()

    def _worker():
        try:
            for node_name, node_output in _stream_sync(query, thread_id):
                q.put((node_name, node_output))
            q.put(None)  # sentinel
        except Exception as e:
            q.put(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        # Poll queue without blocking the event loop
        while q.empty():
            await asyncio.sleep(0.1)

        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item