"""Token and cost tracking for LLM calls.

Tracks per-request and cumulative token usage across agent nodes.
Uses LangChain's callback mechanism to intercept usage metadata.
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass, field
from core.logger import get_logger

logger = get_logger(__name__)

# Groq pricing per 1M tokens (as of 2026 — adjust as needed)
MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}


@dataclass
class RequestMetrics:
    """Metrics for a single API request."""
    thread_id: str = ""
    query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    agent_metrics: Dict[str, Dict] = field(default_factory=dict)
    latency_ms: float = 0.0

    def add_agent_usage(self, agent: str, input_tokens: int, output_tokens: int, model: str = ""):
        """Record token usage for a specific agent node."""
        self.agent_metrics[agent] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
        }
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens

    def calculate_cost(self, model: str):
        """Estimate cost based on model pricing."""
        pricing = MODEL_PRICING.get(model, {"input": 0.59, "output": 0.79})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        self.estimated_cost_usd = round(input_cost + output_cost, 6)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "thread_id": self.thread_id,
            "latency_ms": self.latency_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "agent_breakdown": self.agent_metrics,
        }


class UsageTracker:
    """Cumulative usage tracker across all requests. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.recent_requests: list = []  # Last 50 requests

    def record(self, metrics: RequestMetrics):
        """Record a completed request's metrics."""
        with self._lock:
            self.total_requests += 1
            self.total_input_tokens += metrics.total_input_tokens
            self.total_output_tokens += metrics.total_output_tokens
            self.total_tokens += metrics.total_tokens
            self.total_cost_usd += metrics.estimated_cost_usd
            self.recent_requests.append(metrics.to_dict())
            self.recent_requests = self.recent_requests[-50:]

            logger.info(
                f"Usage recorded | request #{self.total_requests} "
                f"tokens={metrics.total_tokens} cost=${metrics.estimated_cost_usd}"
            )

    def get_summary(self) -> dict:
        """Get cumulative usage summary."""
        with self._lock:
            avg_tokens = (
                self.total_tokens / self.total_requests
                if self.total_requests > 0
                else 0
            )
            return {
                "total_requests": self.total_requests,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "avg_tokens_per_request": round(avg_tokens),
                "recent_requests": self.recent_requests[-10:],
            }

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self.total_requests = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_tokens = 0
            self.total_cost_usd = 0.0
            self.recent_requests = []
            logger.info("Usage tracker reset")


# Global singleton
usage_tracker = UsageTracker()
