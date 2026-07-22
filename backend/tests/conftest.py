"""Shared test fixtures for agent unit tests."""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from agents import tools as agent_tools
from core import usage as core_usage


@pytest.fixture(autouse=True)
def _isolate_collectors():
    """Reset the thread-local source and token collectors between tests.

    They accumulate across a run by design (so a retry pass adds to the first
    pass), which would otherwise leak state from one test into the next.
    """
    agent_tools.reset_sources()
    core_usage.reset_llm_usage()
    yield


@pytest.fixture
def sample_state():
    """Return a clean initial AgentState dict for testing."""
    return {
        "messages": [HumanMessage(content="test query")],
        "query": "What are the latest trends in AI?",
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


@pytest.fixture
def sample_research_output():
    """Return a realistic researcher output string."""
    return """FINDINGS:
1. Large language models have seen significant adoption in enterprise applications in 2026.
2. Multi-agent systems are becoming the dominant paradigm for complex AI tasks.
3. [UNCERTAIN] Some reports suggest AGI timelines have shifted.

SOURCES:
- AI industry reports 2026
- Recent multi-agent research papers
"""


@pytest.fixture
def sample_analysis_output():
    """Return a realistic analyst output string."""
    return """KEY INSIGHTS:
1. Enterprise LLM adoption has grown 300% year-over-year.
2. Multi-agent architectures outperform single-agent approaches on complex tasks.

DATA POINTS:
- 300% YoY growth in enterprise LLM deployments
- 85% of Fortune 500 companies using AI agents

GAPS IDENTIFIED:
- Missing specific market size data
"""


@pytest.fixture
def mock_llm_response():
    """Return a mock LLM response object."""
    mock = MagicMock()
    mock.content = "Mocked LLM response content"
    mock.tool_calls = []
    return mock
