"""Unit tests for the writer agent."""

import pytest
from unittest.mock import patch, MagicMock
from agents.writer import writer_node
from schemas.models import WriterReport


class TestWriterNode:
    """Tests for writer_node function."""

    @patch("agents.writer.get_writer_llm")
    def test_successful_report_generation(self, mock_llm_factory, sample_state, sample_research_output, sample_analysis_output):
        """Writer should produce a structured report dict."""
        sample_state["research_output"] = sample_research_output
        sample_state["analysis_output"] = sample_analysis_output
        sample_state["confidence"] = 0.8

        mock_report = WriterReport(
            title="AI Trends Report 2026",
            summary="AI adoption is accelerating across industries.",
            research_findings=["LLMs growing", "Multi-agent systems emerging"],
            analysis=["300% YoY growth"],
            conclusion="AI is transforming enterprise operations.",
        )

        mock_llm = MagicMock()
        structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured_llm

        chain = MagicMock()
        chain.invoke.return_value = mock_report
        structured_llm.__or__ = MagicMock(return_value=chain)

        with patch("agents.writer.WRITER_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=chain)
            mock_llm_factory.return_value = mock_llm

            result = writer_node(sample_state)

        assert result["final_report"] is not None
        assert isinstance(result["final_report"], dict)
        assert result["final_report"]["title"] == "AI Trends Report 2026"
        assert result["next_agent"] == "END"
        assert result["iterations"] == 1

    @patch("agents.writer.get_writer_llm")
    def test_low_confidence_flags_review(self, mock_llm_factory, sample_state):
        """Writer should flag for human review when confidence < threshold."""
        sample_state["confidence"] = 0.3  # Below default threshold of 0.7

        mock_report = WriterReport(
            title="Low Confidence Report",
            summary="Limited data available.",
            research_findings=["Sparse data"],
            analysis=["Inconclusive"],
            conclusion="More research needed.",
        )

        mock_llm = MagicMock()
        structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured_llm

        chain = MagicMock()
        chain.invoke.return_value = mock_report
        structured_llm.__or__ = MagicMock(return_value=chain)

        with patch("agents.writer.WRITER_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=chain)
            mock_llm_factory.return_value = mock_llm

            result = writer_node(sample_state)

        assert result["needs_human_review"] is True
        assert "below threshold" in result["review_reason"].lower()

    @patch("agents.writer.get_writer_llm")
    def test_llm_error_produces_fallback_report(self, mock_llm_factory, sample_state):
        """On LLM error, writer should produce a degraded fallback report dict."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured_llm

        chain = MagicMock()
        chain.invoke.side_effect = Exception("Structured output parsing failed")
        structured_llm.__or__ = MagicMock(return_value=chain)

        with patch("agents.writer.WRITER_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=chain)
            mock_llm_factory.return_value = mock_llm

            result = writer_node(sample_state)

        assert result["final_report"] is not None
        assert result["final_report"]["title"] == "Report Generation Failed"
        assert result["confidence"] == 0.0
        assert result["needs_human_review"] is True
