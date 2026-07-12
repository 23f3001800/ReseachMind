"""Search tools for the researcher agent.

The researcher LLM decides when and what to search via tool calling,
making it a true ReAct agent rather than a fixed chain.

Supports DuckDuckGo (free, no key) and Tavily (better quality, needs API key).
The active tool is selected by the `search_provider` setting in config.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from core.logger import get_logger

logger = get_logger(__name__)


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


@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily for high-quality, AI-optimized results.
    Use this tool to find factual, up-to-date information.
    You can call this tool multiple times with different queries
    to gather comprehensive information.

    Args:
        query: The search query string.
    """
    try:
        from langchain_tavily import TavilySearch
        from config import settings

        search = TavilySearch(
            max_results=5,
            api_key=settings.tavily_api_key,
        )
        results = search.invoke(query)

        # TavilySearch returns a list of dicts or a string
        if isinstance(results, list):
            formatted = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                formatted.append(f"{title}: {content} ({url})")
            return "\n\n".join(formatted) if formatted else "No results found."
        return str(results) if results else "No results found for this query."
    except Exception as e:
        return f"Tavily search failed: {str(e)}. Rely on your internal knowledge."


def get_search_tools() -> list:
    """Return the appropriate search tool based on config.

    - 'auto': use Tavily if API key is set, otherwise DuckDuckGo
    - 'tavily': use Tavily (requires TAVILY_API_KEY)
    - 'duckduckgo': use DuckDuckGo (free, no key needed)
    """
    from config import settings

    provider = settings.search_provider

    if provider == "tavily" or (provider == "auto" and settings.tavily_api_key):
        logger.info("Search provider: Tavily")
        return [tavily_search]
    else:
        logger.info("Search provider: DuckDuckGo")
        return [web_search]
