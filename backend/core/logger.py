"""Logging factory.

Emits JSON when LOG_FORMAT=json, plain text otherwise.

The distinction matters in a hosted environment: Azure Container Apps forwards
stdout to Log Analytics, which indexes JSON fields but treats a pipe-delimited
line as one opaque string. Without this you cannot filter by severity, group
errors, or query by thread_id — you can only grep.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Attributes present on every LogRecord; anything else was passed via `extra`
# and is therefore application context worth emitting as a field.
_STANDARD = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})

# Python level names -> the severity strings Azure Monitor / Cloud Logging expect.
_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "severity": _SEVERITY.get(record.levelname, record.levelname),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Structured context from logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    return handler


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger.

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Pipeline finished", extra={"thread_id": tid, "tokens": n})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.addHandler(_build_handler())
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        # Handlers are attached per-logger here, so let the root logger alone
        # rather than emitting every record twice.
        logger.propagate = False

    return logger
