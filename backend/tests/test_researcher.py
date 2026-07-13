"""Unit tests for the researcher agent."""

from unittest.mock import patch, MagicMock
from agents.researcher import researcher_node


class TestResearcherNode:
    """Tests for researcher_node function."""

    @patch("agents.researcher.get_search_tools")
    @patch("agents.researcher.get_researcher_llm")
    def test_successful_research(self, mock_llm_factory, mock_tools, sample_state):
        """Researcher should return research output, sources, and confidence."""
        # Mock the search tools
        mock_search = MagicMock()
        mock_search.name = "web_search"
        mock_search.invoke.return_value = "Search result text"
        mock_tools.return_value = [mock_search]

        # Mock the LLM to NOT call tools (final response)
        mock_response = MagicMock()
        mock_response.content = """FINDINGS:
1. AI is advancing rapidly.
2. Multi-agent systems are growing.

SOURCES:
- AI research papers 2026
- Industry reports"""
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        result = researcher_node(sample_state)

        assert result["research_output"] is not None
        assert result["confidence"] == 0.8
        assert result["next_agent"] == "analyst"
        assert len(result["sources"]) > 0
        assert result["iterations"] == 1

    @patch("agents.researcher.get_search_tools")
    @patch("agents.researcher.get_researcher_llm")
    def test_uncertain_research_lowers_confidence(self, mock_llm_factory, mock_tools, sample_state):
        """Confidence should drop to 0.5 when [UNCERTAIN] is in the output."""
        mock_search = MagicMock()
        mock_search.name = "web_search"
        mock_tools.return_value = [mock_search]

        mock_response = MagicMock()
        mock_response.content = "FINDINGS:\n1. [UNCERTAIN] Data is limited.\n\nSOURCES:\n- unknown"
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        result = researcher_node(sample_state)

        assert result["confidence"] == 0.5

    @patch("agents.researcher.get_search_tools")
    @patch("agents.researcher.get_researcher_llm")
    def test_llm_failure_flags_for_review(self, mock_llm_factory, mock_tools, sample_state):
        """On LLM error, researcher should flag for human review and skip to writer."""
        mock_tools.return_value = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("Groq API error")
        mock_llm_factory.return_value = mock_llm

        result = researcher_node(sample_state)

        assert result["confidence"] == 0.2
        assert result["needs_human_review"] is True
        assert result["next_agent"] == "writer"
        assert "error" in result["review_reason"].lower()
