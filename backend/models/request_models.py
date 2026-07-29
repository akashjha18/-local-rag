"""
request_models.py — API Request/Response Schemas
=================================================
Pydantic models for FastAPI request validation and
response serialization.

These are the API boundary models — completely separate from
the internal document_models.py dataclasses.

Why separate?
  Internal models (dataclasses) → optimized for Python speed
  API models (Pydantic) → optimized for JSON validation + docs
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request body for POST /api/v1/query"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer from indexed documents",
        examples=["What is the main finding of this research?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0=all, 1=exact match)",
    )
    filter_document_id: Optional[str] = Field(
        default=None,
        description="Limit search to a specific document ID",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM creativity (0=deterministic, 1=creative)",
    )


class SourceResponse(BaseModel):
    """A single source citation in the query response."""
    filename: str
    page_number: int
    confidence_percent: int
    document_id: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Response body for POST /api/v1/query"""
    query: str
    answer: str
    sources: list[SourceResponse]
    model: str
    chunks_retrieved: int
    retrieval_time: float
    generation_time: float
    total_time: float
    success: bool
    error: Optional[str] = None


class DocumentResponse(BaseModel):
    """A document entry in the index."""
    document_id: str
    filename: str
    document_type: str
    chunk_count: int


class IngestResponse(BaseModel):
    """Response body for POST /api/v1/documents"""
    document_id: str
    filename: str
    chunks_indexed: int
    page_count: int
    word_count: int
    processing_time: float
    status: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status: str
    pipeline_ready: bool
    llm_available: bool
    llm_model: str
    available_models: list[str]
    embedding_model: str
    total_vectors: int
    indexed_documents: int
    index_size_mb: float


class StatsResponse(BaseModel):
    """Response body for GET /api/v1/stats"""
    total_vectors: int
    index_size_mb: float
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    indexed_documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    """Response body for DELETE /api/v1/documents/{id}"""
    document_id: str
    chunks_removed: int
    success: bool
    message: str