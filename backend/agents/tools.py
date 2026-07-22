"""Search tools for the researcher agent.

The researcher LLM decides when and what to search via tool calling,
making it a true ReAct agent rather than a fixed chain.

Supports DuckDuckGo (free, no key) and Tavily (better quality, needs API key).
The active tool is selected by the `search_provider` setting in config.

Source provenance
-----------------
Tools return a human-readable string to the LLM (that is the tool-calling
contract), but they *also* record the structured results — real URLs and
titles — into a thread-local collector. The researcher reads sources from
that collector, never from the model's own prose, so a report can only ever
cite pages the system actually retrieved.
"""

import os
import threading
from typing import Dict, List

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from core.logger import get_logger

logger = get_logger(__name__)

# Thread-local so concurrent requests never mix sources. The graph runs
# inside a worker thread (see core.supervisor), so reset/read must happen
# on that same thread.
_local = threading.local()


def reset_sources() -> None:
    """Start a fresh source collection for the current thread."""
    _local.sources = []


def collect_source(url: str, title: str = "", provider: str = "") -> None:
    """Record a retrieved source, de-duplicated by URL."""
    if not url:
        return
    sources: List[Dict] = getattr(_local, "sources", None)
    if sources is None:
        sources = _local.sources = []
    if any(s["url"] == url for s in sources):
        return
    sources.append({
        "url": url,
        "title": (title or url)[:200],
        "provider": provider,
    })


def get_collected_sources() -> List[Dict]:
    """Return the sources retrieved so far on this thread."""
    return list(getattr(_local, "sources", []))


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
        search = DuckDuckGoSearchResults(output_format="list", max_results=5)
        results = search.invoke(query)

        if not results:
            return "No results found for this query."

        formatted = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("link", "")
            collect_source(url, title, provider="duckduckgo")
            formatted.append(f"{title}: {snippet} ({url})")

        logger.info(f"DuckDuckGo search | query='{query[:60]}' results={len(formatted)}")
        return "\n\n".join(formatted)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed | query='{query[:60]}' error={e}")
        return f"Search failed: {str(e)}. Rely on your internal knowledge."


def _tavily_ready() -> bool:
    """True if Tavily is configured and importable.

    langchain-tavily has no `api_key` field — it reads TAVILY_API_KEY from the
    process environment. pydantic-settings loads .env into the Settings object
    but never exports it, so the variable has to be published here or every
    Tavily call fails validation and silently degrades to zero sources.
    """
    from config import settings

    if not settings.tavily_api_key:
        return False
    os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
    try:
        from langchain_tavily import TavilySearch  # noqa: F401
        return True
    except ImportError:
        logger.warning("TAVILY_API_KEY is set but langchain-tavily is not installed")
        return False


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

        _tavily_ready()  # publishes TAVILY_API_KEY for the client
        search = TavilySearch(max_results=5)
        results = search.invoke(query)

        # TavilySearch returns either a list of dicts or a dict with "results"
        if isinstance(results, dict):
            results = results.get("results", [])

        if isinstance(results, list):
            formatted = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                collect_source(url, title, provider="tavily")
                formatted.append(f"{title}: {content} ({url})")

            logger.info(f"Tavily search | query='{query[:60]}' results={len(formatted)}")
            return "\n\n".join(formatted) if formatted else "No results found."

        return str(results) if results else "No results found for this query."
    except Exception as e:
        logger.warning(f"Tavily search failed | query='{query[:60]}' error={e}")
        return f"Tavily search failed: {str(e)}. Rely on your internal knowledge."


def get_search_tools() -> list:
    """Return the appropriate search tool based on config.

    - 'auto': use Tavily if it is genuinely usable, otherwise DuckDuckGo
    - 'tavily': use Tavily (requires a working TAVILY_API_KEY)
    - 'duckduckgo': use DuckDuckGo (free, no key needed)

    'auto' checks that Tavily can actually run, not merely that a key string is
    present — selecting a provider that then fails on every call produces a
    report with zero citations, which is the worst outcome this system can have.
    """
    from config import settings

    provider = settings.search_provider

    if provider == "tavily":
        if not _tavily_ready():
            logger.error("SEARCH_PROVIDER=tavily but Tavily is unusable — falling back to DuckDuckGo")
            return [web_search]
        logger.info("Search provider: Tavily")
        return [tavily_search]

    if provider == "auto" and _tavily_ready():
        logger.info("Search provider: Tavily (auto)")
        return [tavily_search]

    logger.info("Search provider: DuckDuckGo")
    return [web_search]
