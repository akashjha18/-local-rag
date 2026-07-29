"""
health.py — Health Check Endpoint
===================================
GET /health — Returns system status.
Used by Docker healthchecks and monitoring tools.
"""

from fastapi import APIRouter, Request
from backend.models.request_models import HealthResponse
from backend.utils.logger import logger

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns the status of all pipeline components.",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Check if the RAG pipeline is ready to serve requests.

    Returns status of:
    - Pipeline initialization
    - Ollama LLM availability
    - Embedding model status
    - Vector index statistics
    """
    pipeline = request.app.state.pipeline
    stats = pipeline.get_stats()

    logger.debug("Health check called")

    return HealthResponse(
        status="ok" if stats["llm_available"] else "degraded",
        pipeline_ready=stats["pipeline_ready"],
        llm_available=stats["llm_available"],
        llm_model=stats["llm_model"],
        available_models=stats["available_models"],
        embedding_model=stats["embedding_model"],
        total_vectors=stats["total_vectors"],
        indexed_documents=len(stats["indexed_documents"]),
        index_size_mb=round(stats["index_size_mb"], 3),
    )