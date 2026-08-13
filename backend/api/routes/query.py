"""
query.py — Question Answering Endpoints
=======================================
POST /api/v1/query  — Ask a question
GET  /api/v1/models — List Ollama models
GET  /api/v1/stats  — System statistics
"""

from fastapi import APIRouter, HTTPException, Request

from backend.models.request_models import (
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    StatsResponse,
)
from backend.utils.logger import logger
from backend.utils.query_processor import preprocess_query

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a Question",
    description="Query indexed documents using natural language.",
)
async def query_documents(
    body: QueryRequest,
    request: Request,
) -> QueryResponse:
    """
    Answer a question using the RAG pipeline.

    1. Preprocess and validate the query
    2. Embed the query
    3. Search FAISS for relevant chunks
    4. Send chunks + query to Ollama LLM
    5. Return answer with source citations
    """
    pipeline = request.app.state.pipeline

    # ── Check index has documents ──────────────────────────────────
    if pipeline.vector_store.is_empty:
        raise HTTPException(
            status_code=400,
            detail=(
                "No documents indexed. "
                "Please upload documents first via POST /api/v1/documents"
            ),
        )

    # ── Preprocess query ───────────────────────────────────────────
    try:
        clean_query = preprocess_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Query received: '{clean_query[:60]}'")

    # ── Run RAG pipeline ───────────────────────────────────────────
    response = pipeline.ask(
        query=clean_query,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        filter_document_id=body.filter_document_id,
        temperature=body.temperature,
    )

    # ── Build source response objects ──────────────────────────────
    sources = [
        SourceResponse(
            filename=src.filename,
            page_number=src.page_number,
            confidence_percent=src.confidence_percent,
            document_id=src.document_id,
            chunk_index=src.chunk_index,
        )
        for src in response.sources
    ]

    return QueryResponse(
        query=clean_query,
        answer=response.answer,
        sources=sources,
        model=response.model,
        chunks_retrieved=response.chunks_retrieved,
        retrieval_time=round(response.retrieval_time, 3),
        generation_time=round(response.generation_time, 3),
        total_time=round(response.total_time, 3),
        success=response.success,
        error=response.error,
    )


@router.get(
    "/models",
    summary="List Available Models",
    description="Returns locally available Ollama models.",
)
async def list_models(request: Request) -> dict:
    """List all Ollama models available on this machine."""
    pipeline = request.app.state.pipeline
    models = pipeline.llm.list_models()

    return {
        "models": models,
        "current_model": pipeline.llm.model,
        "total": len(models),
    }


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System Statistics",
    description="Detailed statistics about the RAG system state.",
)
async def get_stats(request: Request) -> StatsResponse:
    """Return detailed system statistics."""
    pipeline = request.app.state.pipeline
    stats = pipeline.get_stats()

    indexed_docs = [
        DocumentResponse(
            document_id=d["document_id"],
            filename=d["filename"],
            document_type=d["document_type"],
            chunk_count=d["chunk_count"],
        )
        for d in stats["indexed_documents"]
    ]

    return StatsResponse(
        total_vectors=stats["total_vectors"],
        index_size_mb=round(stats["index_size_mb"], 3),
        embedding_model=stats["embedding_model"],
        llm_model=stats["llm_model"],
        chunk_size=stats["chunk_size"],
        chunk_overlap=stats["chunk_overlap"],
        top_k=stats["top_k"],
        score_threshold=stats["score_threshold"],
        indexed_documents=indexed_docs,
    )