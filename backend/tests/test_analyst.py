"""Unit tests for the analyst agent."""

from unittest.mock import patch, MagicMock
from agents.analyst import analyst_node


class TestAnalystNode:
    """Tests for analyst_node function."""

    @patch("agents.analyst.get_analyst_llm")
    def test_successful_analysis(self, mock_llm_factory, sample_state, sample_research_output):
        """Analyst should extract insights and pass to writer."""
        sample_state["research_output"] = sample_research_output

        mock_response = MagicMock()
        mock_response.content = """KEY INSIGHTS:
1. Enterprise AI adoption is accelerating.

DATA POINTS:
- 300% YoY growth

GAPS IDENTIFIED:
- None significant"""

        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response.content
        mock_llm.__or__ = MagicMock(return_value=mock_chain)
        mock_llm_factory.return_value = mock_llm

        # Patch the chain directly
        with patch("agents.analyst.StrOutputParser") as mock_parser:
            mock_parser_instance = MagicMock()
            mock_parser.return_value = mock_parser_instance

            with patch("agents.analyst.ANALYST_PROMPT") as mock_prompt:
                chain = MagicMock()
                chain.invoke.return_value = mock_response.content
                mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))

                result = analyst_node(sample_state)

        assert result["analysis_output"] is not None
        assert result["next_agent"] in ("writer", "researcher")
        assert result["iterations"] == 1

    def test_empty_research_flags_review(self, sample_state):
        """Analyst should flag for review when research output is empty."""
        sample_state["research_output"] = ""

        result = analyst_node(sample_state)

        assert result["confidence"] == 0.1
        assert result["needs_human_review"] is True
        assert result["next_agent"] == "writer"
        assert "empty" in result["review_reason"].lower()

    def test_none_research_flags_review(self, sample_state):
        """Analyst should flag for review when research output is None."""
        sample_state["research_output"] = None

        result = analyst_node(sample_state)

        assert result["confidence"] == 0.1
        assert result["needs_human_review"] is True

    @patch("agents.analyst.get_analyst_llm")
    def test_low_confidence_flag(self, mock_llm_factory, sample_state, sample_research_output):
        """[LOW-CONFIDENCE] in output should cap confidence at 0.5."""
        sample_state["research_output"] = sample_research_output
        sample_state["confidence"] = 0.9

        with patch("agents.analyst.ANALYST_PROMPT") as mock_prompt:
            chain = MagicMock()
            chain.invoke.return_value = "KEY INSIGHTS:\n1. [LOW-CONFIDENCE] Unclear data\n\nDATA POINTS:\n- None\n\nGAPS IDENTIFIED:\n- None"
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))

            mock_llm_factory.return_value = MagicMock()
            result = analyst_node(sample_state)

        assert result["confidence"] <= 0.5

    @patch("agents.analyst.get_analyst_llm")
    def test_llm_error_flags_review(self, mock_llm_factory, sample_state, sample_research_output):
        """On LLM error, analyst should flag for human review."""
        sample_state["research_output"] = sample_research_output

        with patch("agents.analyst.ANALYST_PROMPT") as mock_prompt:
            chain = MagicMock()
            chain.invoke.side_effect = Exception("LLM timeout")
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))

            mock_llm_factory.return_value = MagicMock()
            result = analyst_node(sample_state)

        assert result["confidence"] == 0.2
        assert result["needs_human_review"] is True
        assert result["next_agent"] == "writer"
