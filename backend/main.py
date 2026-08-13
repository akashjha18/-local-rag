"""
main.py — FastAPI Application Entry Point
==========================================
Creates the FastAPI app, registers all routes,
and manages the RAGPipeline singleton lifecycle.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.api.routes import health, documents, query
from backend.api.middleware.cors import setup_cors
from backend.api.middleware.logging_middleware import LoggingMiddleware
from backend.rag.pipeline import RAGPipeline
from backend.config import get_settings
from backend.utils.logger import logger, setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — startup and shutdown logic.
    Everything before yield runs at startup.
    Everything after yield runs at shutdown.
    """
    # ── STARTUP ────────────────────────────────────────────────────
    settings = get_settings()
    setup_logger(debug=settings.debug)

    logger.info("=" * 50)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 50)

    logger.info("Loading RAG Pipeline (this takes a few seconds)...")
    app.state.pipeline = RAGPipeline()

    logger.info("Server ready — accepting requests")

    yield

    # ── SHUTDOWN ───────────────────────────────────────────────────
    logger.info("Shutting down — saving FAISS index...")
    try:
        app.state.pipeline.vector_store.save()
        logger.info("FAISS index saved successfully")
    except Exception as e:
        logger.error(f"Error saving index on shutdown: {e}")

    logger.info("Server shutdown complete")


# ── Create FastAPI app ─────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Local RAG System — AI-powered document search that runs "
        "completely offline. Upload PDFs and DOCX files, then ask "
        "questions in natural language."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Register middleware ────────────────────────────────────────────
setup_cors(app)
app.add_middleware(LoggingMiddleware)


# ── Global exception handlers ──────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Return clean error messages for validation failures.
    Instead of FastAPI's verbose default, return a simple message.
    """
    errors = exc.errors()
    messages = []

    for error in errors:
        field = " → ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        messages.append(f"{field}: {msg}")

    logger.warning(f"Validation error: {'; '.join(messages)}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": messages,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Check server logs.",
        },
    )


# ── Register routers ───────────────────────────────────────────────
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


# ── Root endpoint ──────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    })