"""Integration tests for the FastAPI endpoints."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "agentic-research-assistant"


class TestGraphEndpoint:
    """Tests for GET /agent/graph."""

    def test_graph_returns_structure(self):
        response = client.get("/agent/graph")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "researcher" in data["agents"]
        assert "analyst" in data["agents"]
        assert "writer" in data["agents"]
        assert "guardrails" in data


class TestChatEndpoint:
    """Tests for POST /agent/chat."""

    def test_empty_message_returns_400(self):
        response = client.post("/agent/chat", json={
            "message": "   ",
            "thread_id": "test",
        })
        assert response.status_code == 400

    def test_short_message_returns_422(self):
        """Pydantic min_length=3 validation should reject tiny messages."""
        response = client.post("/agent/chat", json={
            "message": "ab",
            "thread_id": "test",
        })
        assert response.status_code == 422

    @patch("main.run_agent", new_callable=AsyncMock)
    def test_successful_chat(self, mock_run_agent):
        """Full pipeline should return a structured ChatResponse."""
        mock_run_agent.return_value = {
            "final_report": {
                "title": "Test Report",
                "summary": "A test summary.",
                "research_findings": ["Finding 1", "Finding 2"],
                "analysis": ["Insight 1"],
                "conclusion": "Test conclusion.",
            },
            "sources": ["source1.com"],
            "confidence": 0.85,
            "needs_human_review": False,
            "iterations": 3,
        }

        response = client.post("/agent/chat", json={
            "message": "What are the latest AI trends?",
            "thread_id": "test-thread",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-thread"
        assert data["report"]["title"] == "Test Report"
        assert data["report"]["confidence"] == 0.85
        assert data["iterations"] == 3
        assert data["latency_ms"] > 0

    @patch("main.run_agent", new_callable=AsyncMock)
    def test_no_output_returns_500(self, mock_run_agent):
        """Should return 500 when pipeline produces no report."""
        mock_run_agent.return_value = {
            "final_report": None,
            "sources": [],
            "confidence": 0.0,
        }

        response = client.post("/agent/chat", json={
            "message": "Test query with no output",
            "thread_id": "test",
        })

        assert response.status_code == 500

    @patch("main.run_agent", new_callable=AsyncMock)
    def test_pipeline_exception_returns_500(self, mock_run_agent):
        """Should return 500 when pipeline raises an exception."""
        mock_run_agent.side_effect = Exception("LLM provider is down")

        response = client.post("/agent/chat", json={
            "message": "Test query with error",
            "thread_id": "test",
        })

        assert response.status_code == 500


class TestHistoryEndpoints:
    """Tests for GET/DELETE /agent/history/{thread_id}."""

    @patch("main.get_conversation_history")
    def test_get_empty_history(self, mock_history):
        mock_history.return_value = []

        response = client.get("/agent/history/test-thread")

        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-thread"
        assert data["count"] == 0
        assert data["exchanges"] == []

    @patch("main.get_conversation_history")
    def test_get_history_with_data(self, mock_history):
        mock_history.return_value = [
            {"query": "AI trends", "report": "Summary of AI trends"},
            {"query": "ML frameworks", "report": "Summary of ML frameworks"},
        ]

        response = client.get("/agent/history/test-thread")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

    @patch("main.clear_thread")
    def test_clear_history(self, mock_clear):
        response = client.delete("/agent/history/test-thread")

        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()
        mock_clear.assert_called_once_with("test-thread")
