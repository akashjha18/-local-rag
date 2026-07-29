"""
main.py — FastAPI Application Entry Point
==========================================
Creates the FastAPI app, registers all routes,
and manages the RAGPipeline singleton lifecycle.

Key pattern: lifespan context manager
  - Runs startup code ONCE before accepting requests
  - Runs shutdown code ONCE after last request
  - Stores shared state in app.state

This ensures:
  - RAGPipeline (embedding model + FAISS) loads once
  - Every request shares the same loaded components
  - Graceful shutdown saves the FAISS index
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.api.routes import health, documents, query
from backend.api.middleware.cors import setup_cors
from backend.rag.pipeline import RAGPipeline
from backend.config import get_settings
from backend.utils.logger import logger, setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything BEFORE yield runs at startup.
    Everything AFTER yield runs at shutdown.

    Why lifespan instead of @app.on_event?
    lifespan is the modern FastAPI pattern (v0.93+).
    on_event is deprecated.
    """
    # ── STARTUP ────────────────────────────────────────────────────
    settings = get_settings()
    setup_logger(debug=settings.debug)

    logger.info("=" * 50)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 50)

    # Initialize the RAG Pipeline — loads models, opens FAISS index
    # This runs ONCE and the pipeline is shared across ALL requests
    logger.info("Loading RAG Pipeline (this takes a few seconds)...")
    app.state.pipeline = RAGPipeline()

    logger.info("Server ready — accepting requests")

    yield  # ← Server runs here, handling requests

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
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI at http://localhost:8000/redoc
    lifespan=lifespan,
)

# ── Register middleware ────────────────────────────────────────────
setup_cors(app)

# ── Register routers ───────────────────────────────────────────────
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


# ── Root endpoint ──────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return JSONResponse({
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    })