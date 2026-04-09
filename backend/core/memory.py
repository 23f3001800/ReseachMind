from langgraph.checkpoint.memory import MemorySaver
from typing import Dict

# In-memory checkpointer — stores conversation state per thread_id
# In production: replace with SqliteSaver or PostgresSaver
_checkpointer = None
_conversation_store: Dict[str, list] = {}


def get_checkpointer() -> MemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def get_conversation_history(thread_id: str) -> list:
    return _conversation_store.get(thread_id, [])


def save_to_history(thread_id: str, query: str, report: str):
    if thread_id not in _conversation_store:
        _conversation_store[thread_id] = []
    _conversation_store[thread_id].append({
        "query": query,
        "report": report,
    })
    # Keep last 20 exchanges per thread
    _conversation_store[thread_id] = _conversation_store[thread_id][-20:]


def clear_thread(thread_id: str):
    if thread_id in _conversation_store:
        del _conversation_store[thread_id]