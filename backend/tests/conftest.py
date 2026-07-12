"""Shared test fixtures for agent unit tests."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage


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
        "retry_count": 0,
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
