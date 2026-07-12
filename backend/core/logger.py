import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger with structured format.

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("message", extra={"key": "value"})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
