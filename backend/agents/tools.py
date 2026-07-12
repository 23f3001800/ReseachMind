"""Search tools for the researcher agent.

The researcher LLM decides when and what to search via tool calling,
making it a true ReAct agent rather than a fixed chain.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.
    Use this tool to find factual, up-to-date information.
    You can call this tool multiple times with different queries
    to gather comprehensive information.

    Args:
        query: The search query string.
    """
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results if results else "No results found for this query."
    except Exception as e:
        return f"Search failed: {str(e)}. Rely on your internal knowledge."
