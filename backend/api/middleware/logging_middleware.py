"""
logging_middleware.py — Request/Response Logging
=================================================
Logs every API request with timing information.
Essential for debugging and monitoring in production.

Logs format:
  → POST /api/v1/query  (from 127.0.0.1)
  ← 200 OK  [18.3s]
"""

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from backend.utils.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that logs all HTTP requests and responses.

    Middleware wraps every request:
    1. Log incoming request (method, path, client IP)
    2. Process the request normally
    3. Log the response (status code, timing)

    Usage in main.py:
        app.add_middleware(LoggingMiddleware)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Called for every HTTP request.

        Args:
            request:   The incoming HTTP request
            call_next: Function to call the actual route handler

        Returns:
            The HTTP response
        """
        start_time = time.time()

        # ── Log incoming request ───────────────────────────────────
        client_ip = request.client.host if request.client else "unknown"

        # Skip logging for health checks to reduce noise
        is_health_check = request.url.path in ("/health", "/")

        if not is_health_check:
            logger.info(
                f"→ {request.method} {request.url.path} "
                f"(from {client_ip})"
            )

        # ── Process request ────────────────────────────────────────
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unhandled exceptions
            logger.error(
                f"✗ {request.method} {request.url.path} "
                f"— Unhandled error: {e}"
            )
            raise

        # ── Log response ───────────────────────────────────────────
        elapsed = time.time() - start_time

        if not is_health_check:
            # Choose log level based on status code
            if response.status_code >= 500:
                log_fn = logger.error
            elif response.status_code >= 400:
                log_fn = logger.warning
            else:
                log_fn = logger.info

            log_fn(
                f"← {response.status_code} "
                f"{request.method} {request.url.path} "
                f"[{elapsed:.3f}s]"
            )

        return response