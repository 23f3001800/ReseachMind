import asyncio
import threading
import time
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.state import AgentState
from core.memory import get_checkpointer
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from agents.tools import reset_sources
from core.usage import reset_llm_usage, get_collected_usage, build_metrics_from_usage
from core import events
from core.logger import get_logger
from config import settings

logger = get_logger(__name__)

# Node name for the synthetic event carrying end-of-run telemetry.
FINAL_META_NODE = "__meta__"


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


def _initial_state(query: str) -> AgentState:
    return {
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
        "research_gaps_detail": [],
        "retry_count": 0,
        "token_usage": None,
    }


def _begin_run() -> None:
    """Reset the thread-local collectors for source and token capture.

    Both collectors are thread-local and the whole graph runs on one worker
    thread, so this must be called on that same thread — not from the caller.
    """
    reset_sources()
    reset_llm_usage()


def _collect_token_usage() -> dict:
    """Fold this run's measured LLM calls into a usage dict."""
    metrics = build_metrics_from_usage(get_collected_usage(), settings.llm_model)
    return metrics.to_dict()


def _invoke_sync(query: str, thread_id: str) -> AgentState:
    """Synchronous graph invocation — runs in a thread pool."""
    start = time.perf_counter()
    logger.info(f"Pipeline started | thread_id={thread_id} query='{query[:80]}'")

    _begin_run()
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = dict(graph.invoke(_initial_state(query), config=config))
    result["token_usage"] = _collect_token_usage()

    elapsed = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        f"Pipeline completed | thread_id={thread_id} "
        f"iterations={result.get('iterations', 0)} "
        f"confidence={result.get('confidence', 0)} "
        f"tokens={result['token_usage'].get('total_tokens', 0)} "
        f"duration_ms={elapsed}"
    )
    return result


async def run_agent(query: str, thread_id: str = "default") -> AgentState:
    """Run the full multi-agent pipeline without blocking the async event loop."""
    return await asyncio.to_thread(_invoke_sync, query, thread_id)


def _stream_sync(query: str, thread_id: str):
    """Synchronous generator — yields (node_name, state) tuples as each node completes.

    Ends with a synthetic FINAL_META_NODE event carrying token usage, which is
    only available once every node has run.
    """
    _begin_run()
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream(_initial_state(query), config=config):
        # event is a dict like {"researcher": {state_updates}} or {"analyst": {...}}
        for node_name, node_output in event.items():
            yield node_name, node_output

    yield FINAL_META_NODE, {"token_usage": _collect_token_usage()}


# Node name for in-flight progress reported from inside a node.
PROGRESS_NODE = "__progress__"


async def run_agent_stream(query: str, thread_id: str = "default"):
    """Async generator — yields (node_name, state) tuples without blocking the event loop.

    The graph is synchronous and its collectors are thread-local, so it runs on a
    single worker thread and hands events back through the loop. Results are pushed
    via call_soon_threadsafe rather than polled, so there is no added latency per event.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def _worker():
        # In-flight progress (tool calls, agent starts) is pushed onto the same
        # queue as completed-node events, so the consumer sees one ordered
        # stream rather than having to merge two.
        events.set_sink(
            lambda evt: loop.call_soon_threadsafe(queue.put_nowait, (PROGRESS_NODE, evt))
        )
        try:
            for item in _stream_sync(query, thread_id):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:  # surface to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, e)
        else:
            loop.call_soon_threadsafe(queue.put_nowait, done)
        finally:
            events.set_sink(None)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await queue.get()
        if item is done:
            break
        if isinstance(item, Exception):
            raise item
        yield item
