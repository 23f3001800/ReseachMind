"""In-flight progress events from inside the graph.

LangGraph's stream only yields once a node *completes*, so a researcher doing
five searches over 40 seconds produces no output at all until it finishes — the
UI looks frozen during the most interesting part of the run. This gives nodes a
way to report what they are doing while they do it.

The sink is thread-local, matching how sources and token usage are collected:
the whole graph runs on one worker thread, so a node can emit and the streaming
generator on that same thread picks it up. When nothing is listening (the
non-streaming endpoint, or tests) emit() is a no-op.
"""

import threading
from typing import Any, Callable, Dict, Optional

_local = threading.local()

EventSink = Callable[[Dict[str, Any]], None]


def set_sink(sink: Optional[EventSink]) -> None:
    """Attach a sink for the current thread. Pass None to detach."""
    _local.sink = sink


def emit(event: str, agent: str = "", **data: Any) -> None:
    """Report progress. Silently does nothing if no one is listening.

    Telemetry must never be able to break a pipeline run, so a failing sink is
    swallowed rather than propagated into the agent.
    """
    sink: Optional[EventSink] = getattr(_local, "sink", None)
    if sink is None:
        return
    try:
        sink({"event": event, "agent": agent, "data": data})
    except Exception:
        pass
