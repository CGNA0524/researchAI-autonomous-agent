"""
Structured logging module with automatic secret masking and structured contextual output.
"""

import logging
import re
import sys
from typing import Any, Dict


SECRET_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"sk-proj-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"key-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([\w\-]+)(['\"]?)", re.IGNORECASE),
]


class SafeFormatter(logging.Formatter):
    """Logging formatter that scrubs API keys and sensitive tokens."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern in SECRET_PATTERNS:
            if "api[_-]?key" in pattern.pattern:
                sanitized = pattern.sub(r"\1***REDACTED***\3", sanitized)
            else:
                sanitized = pattern.sub("***REDACTED_API_KEY***", sanitized)
        return sanitized


def get_logger(name: str) -> logging.Logger:
    """Returns a pre-configured logger with secret-masking formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = SafeFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
