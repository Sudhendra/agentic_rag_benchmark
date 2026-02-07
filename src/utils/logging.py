"""Structured logging setup for the benchmark."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO", log_file: str | None = None, format_string: str | None = None
) -> logging.Logger:
    """Set up structured logging for the benchmark.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Create logger
    logger = logging.getLogger("rag_benchmark")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "rag_benchmark") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (dot-separated for hierarchy)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
