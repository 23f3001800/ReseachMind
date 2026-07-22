"""Tests for the in-flight progress event sink and JSON logging."""

import json
import logging

from core import events
from core.logger import JsonFormatter


class TestEventSink:
    def test_emit_without_sink_is_silent(self):
        """The non-streaming path attaches no sink; emitting must not raise."""
        events.set_sink(None)
        events.emit("tool_call", agent="researcher", query="anything")

    def test_emit_reaches_sink(self):
        captured = []
        events.set_sink(captured.append)
        try:
            events.emit("tool_call", agent="researcher", query="vector databases")
        finally:
            events.set_sink(None)

        assert len(captured) == 1
        assert captured[0]["event"] == "tool_call"
        assert captured[0]["agent"] == "researcher"
        assert captured[0]["data"]["query"] == "vector databases"

    def test_failing_sink_never_breaks_the_agent(self):
        """Telemetry must not be able to take down a pipeline run."""
        def broken(_evt):
            raise RuntimeError("sink exploded")

        events.set_sink(broken)
        try:
            events.emit("tool_call", agent="researcher")  # must not raise
        finally:
            events.set_sink(None)

    def test_detaching_stops_delivery(self):
        captured = []
        events.set_sink(captured.append)
        events.set_sink(None)
        events.emit("tool_call", agent="researcher")
        assert captured == []


class TestJsonFormatter:
    def _record(self, **extra):
        record = logging.LogRecord(
            name="core.test", level=logging.INFO, pathname="x.py", lineno=1,
            msg="pipeline finished", args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_emits_valid_json_with_severity(self):
        out = json.loads(JsonFormatter().format(self._record()))
        assert out["message"] == "pipeline finished"
        assert out["severity"] == "INFO"
        assert out["logger"] == "core.test"
        assert "timestamp" in out

    def test_extra_fields_become_queryable_columns(self):
        """The whole point: Log Analytics can filter on these."""
        out = json.loads(JsonFormatter().format(self._record(thread_id="t-1", tokens=1801)))
        assert out["thread_id"] == "t-1"
        assert out["tokens"] == 1801

    def test_non_serialisable_extra_does_not_raise(self):
        out = json.loads(JsonFormatter().format(self._record(obj=object())))
        assert "obj" in out
