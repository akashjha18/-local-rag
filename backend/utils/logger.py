"""
logger.py — Centralized Logging Setup
======================================
Uses Loguru — a modern logging library that's far better than
Python's built-in logging module.

Benefits:
  - Colored terminal output
  - Automatic log rotation
  - Structured log format
  - Simple API: just `from backend.utils.logger import logger`
"""

import sys
from pathlib import Path
from loguru import logger as _logger  # Import as _logger to re-export cleanly


def setup_logger(debug: bool = False) -> None:
    """
    Configure the global logger.
    Called once at application startup from main.py.

    Args:
        debug: If True, show DEBUG level messages. Otherwise INFO+.
    """

    # ── Remove the default Loguru handler ──────────────────────────
    # Loguru adds a default stderr handler. We remove it so we can
    # add our own with custom formatting.
    _logger.remove()

    # ── Log level: DEBUG in dev, INFO in production ────────────────
    log_level = "DEBUG" if debug else "INFO"

    # ── Console Handler ────────────────────────────────────────────
    # colorize=True → Colors in terminal (auto-disabled if redirected)
    # format → What each log line looks like
    _logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " - <level>{message}</level>"
        ),
    )

    # ── File Handler ───────────────────────────────────────────────
    # Also write logs to a rotating file.
    # rotation="10 MB" → Start new file when current hits 10MB
    # retention="7 days" → Keep logs for 7 days then delete
    # compression="zip" → Compress old log files
    log_dir = Path("./data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)  # Create if missing

    _logger.add(
        log_dir / "local_rag_{time:YYYY-MM-DD}.log",
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )

    _logger.info(f"Logger initialized at level: {log_level}")


# ── Export the configured logger ───────────────────────────────────
# Other modules do: from backend.utils.logger import logger
logger = _logger