# agent/graph.py
from langgraph.graph import StateGraph, END
from .state import ResearchState
from .nodes import planner_node, search_node, extract_node, rag_node, synthesise_node

def build_graph(retriever=None):
    """Build and compile the research agent graph."""
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("planner",    planner_node)
    graph.add_node("search",     search_node)
    graph.add_node("extract",    extract_node)
    graph.add_node("rag",        lambda s: rag_node(s, retriever))
    graph.add_node("synthesise", synthesise_node)

    # Linear flow — each node feeds into the next
    graph.set_entry_point("planner")
    graph.add_edge("planner",    "search")
    graph.add_edge("search",     "extract")
    graph.add_edge("extract",    "rag")
    graph.add_edge("rag",        "synthesise")
    graph.add_edge("synthesise", END)

    return graph.compile()


