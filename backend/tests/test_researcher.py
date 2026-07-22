"""Unit tests for the researcher agent."""

from unittest.mock import patch, MagicMock
from agents.researcher import researcher_node
from agents.tools import collect_source


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
2. Multi-agent systems are growing."""
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        result = researcher_node(sample_state)

        assert result["research_output"] is not None
        assert result["confidence"] == 0.8
        assert result["next_agent"] == "analyst"
        assert result["iterations"] == 1

    @patch("agents.researcher.get_search_tools")
    @patch("agents.researcher.get_researcher_llm")
    def test_sources_come_only_from_retrieved_pages(self, mock_llm_factory, mock_tools, sample_state):
        """Sources must reflect pages actually retrieved, not text the model wrote."""
        mock_tools.return_value = []

        # The model invents a citation in its prose...
        mock_response = MagicMock()
        mock_response.content = "FINDINGS:\n1. A claim.\n\nSOURCES:\n- Totally Made Up Journal 2026"
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        # ...while the search layer actually retrieved this page.
        collect_source("https://example.com/real", "Real Page", "duckduckgo")

        result = researcher_node(sample_state)

        assert result["sources"] == [
            {"url": "https://example.com/real", "title": "Real Page", "provider": "duckduckgo"}
        ]
        assert "Totally Made Up Journal" not in str(result["sources"])

    @patch("agents.researcher.get_search_tools")
    @patch("agents.researcher.get_researcher_llm")
    def test_no_searches_means_no_citations(self, mock_llm_factory, mock_tools, sample_state):
        """If the agent never searched, the report cites nothing."""
        mock_tools.return_value = []

        mock_response = MagicMock()
        mock_response.content = "FINDINGS:\n1. From memory.\n\nSOURCES:\n- Some Report"
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        result = researcher_node(sample_state)

        assert result["sources"] == []

    @patch("agents.researcher.get_search_tools")
    @patch("agents.researcher.get_researcher_llm")
    def test_retry_prompt_targets_identified_gaps(self, mock_llm_factory, mock_tools, sample_state):
        """A second pass must ask for the gaps, not repeat the first pass verbatim."""
        mock_tools.return_value = []
        sample_state["research_gaps_detail"] = ["Missing market size data", "No 2026 figures"]
        sample_state["research_output"] = "FINDINGS:\n1. Earlier finding."

        mock_response = MagicMock()
        mock_response.content = "FINDINGS:\n1. Market size is $12B."
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        researcher_node(sample_state)

        sent_messages = mock_llm.invoke.call_args[0][0]
        user_content = sent_messages[1]["content"]
        assert "Missing market size data" in user_content
        assert "No 2026 figures" in user_content
        assert "CLOSE THOSE GAPS" in user_content

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
