"""Tests for gap extraction and source collection.

These two helpers are where the routing and citation bugs lived, so they get
direct coverage rather than only being exercised through the agent nodes.
"""

from agents.analyst import _extract_gaps
from agents.tools import collect_source, get_collected_sources, reset_sources


class TestGapExtraction:
    """Gap detection drives the retry loop, so miscounting costs a full LLM pass."""

    def test_no_section_means_no_gaps(self):
        assert _extract_gaps("KEY INSIGHTS:\n1. Something") == []

    def test_counts_bullet_items(self):
        text = """GAPS IDENTIFIED:
- Missing market size data
- No regional breakdown"""
        assert _extract_gaps(text) == ["Missing market size data", "No regional breakdown"]

    def test_wrapped_gap_is_one_item(self):
        """A single gap spanning two lines must not look like two gaps."""
        text = """GAPS IDENTIFIED:
- The research lacks any figures on enterprise adoption rates
  across the EMEA region for the 2025-2026 period"""
        gaps = _extract_gaps(text)
        assert len(gaps) == 1
        assert "EMEA" in gaps[0]

    def test_none_answers_are_not_gaps(self):
        """"None identified" is the analyst saying there are no gaps."""
        assert _extract_gaps("GAPS IDENTIFIED:\n- None significant") == []
        assert _extract_gaps("GAPS IDENTIFIED:\n- None\n- N/A") == []
        assert _extract_gaps("GAPS IDENTIFIED:\n- Nothing missing") == []

    def test_numbered_items(self):
        text = """GAPS IDENTIFIED:
1. Missing cost data
2. No competitor analysis"""
        assert len(_extract_gaps(text)) == 2

    def test_mixed_none_and_real_gap(self):
        text = """GAPS IDENTIFIED:
- No obvious contradictions
- Missing pricing information"""
        assert _extract_gaps(text) == ["Missing pricing information"]


class TestSourceCollection:
    def test_collects_and_dedupes_by_url(self):
        reset_sources()
        collect_source("https://a.com", "A", "tavily")
        collect_source("https://a.com", "A again", "tavily")
        collect_source("https://b.com", "B", "tavily")

        sources = get_collected_sources()
        assert [s["url"] for s in sources] == ["https://a.com", "https://b.com"]

    def test_ignores_empty_url(self):
        reset_sources()
        collect_source("", "No URL", "duckduckgo")
        assert get_collected_sources() == []

    def test_falls_back_to_url_when_title_missing(self):
        reset_sources()
        collect_source("https://c.com", "", "duckduckgo")
        assert get_collected_sources()[0]["title"] == "https://c.com"

    def test_reset_clears(self):
        reset_sources()
        collect_source("https://d.com", "D", "tavily")
        reset_sources()
        assert get_collected_sources() == []
