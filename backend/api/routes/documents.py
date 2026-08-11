"""
documents.py — Document Management Endpoints
=============================================
POST   /api/v1/documents              → Upload and index a document
GET    /api/v1/documents              → List all indexed documents
DELETE /api/v1/documents/{id}         → Remove a document
POST   /api/v1/documents/{id}/reindex → Re-index a document
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.config import get_settings
from backend.models.request_models import (
    DeleteResponse,
    DocumentResponse,
    IngestResponse,
)
from backend.utils.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 50


@router.post(
    "",
    response_model=IngestResponse,
    summary="Upload and Index Document",
    description="Upload a PDF or DOCX file and index it for search.",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF or DOCX file to upload"),
) -> IngestResponse:
    """
    Upload a document and index it into the RAG system.

    Steps:
    1. Validate file type and size
    2. Save to data/documents/ with UUID prefix
    3. Process with original filename stored in metadata
    4. Store in FAISS index
    5. Return indexing statistics with real filename
    """
    pipeline = request.app.state.pipeline
    settings = get_settings()

    # ── Save the original filename before any modification ─────────
    # This is what we show to users in the UI and source citations
    original_filename = file.filename

    # ── Validate file extension ────────────────────────────────────
    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Read file content ──────────────────────────────────────────
    content = await file.read()

    # ── Validate file size ─────────────────────────────────────────
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    # ── Generate unique document ID ────────────────────────────────
    document_id = str(uuid.uuid4())

    # ── Save file to disk with UUID prefix ─────────────────────────
    # UUID prefix prevents filename collisions when multiple users
    # upload files with the same name
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{document_id}_{original_filename}"
    file_path = docs_dir / safe_filename

    try:
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file: {file_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}",
        )

    # ── Ingest into pipeline ───────────────────────────────────────
    # Pass original_filename so FAISS metadata stores "report.pdf"
    # not "abc123-uuid_report.pdf"
    logger.info(
        f"Starting ingest: {original_filename} "
        f"({size_mb:.2f}MB, id={document_id})"
    )

    result = pipeline.ingest(
        file_path=file_path,
        document_id=document_id,
        original_filename=original_filename,  # ← THE KEY FIX
    )

    if result.status == "failed":
        # Clean up saved file on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Document processing failed: {result.error}",
        )

    logger.info(
        f"Ingest complete: {original_filename} | "
        f"chunks={result.chunks_indexed} | "
        f"time={result.processing_time:.2f}s"
    )

    return IngestResponse(
        document_id=result.document_id,
        filename=original_filename,        # Always return real filename
        chunks_indexed=result.chunks_indexed,
        page_count=result.page_count,
        word_count=result.word_count,
        processing_time=round(result.processing_time, 3),
        status=result.status,
        error=result.error,
    )


@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List Indexed Documents",
    description="Returns all documents currently in the vector index.",
)
async def list_documents(request: Request) -> list[DocumentResponse]:
    """Return all indexed documents with their chunk counts."""
    pipeline = request.app.state.pipeline
    docs = pipeline.get_indexed_documents()

    return [
        DocumentResponse(
            document_id=d["document_id"],
            filename=d["filename"],
            document_type=d["document_type"],
            chunk_count=d["chunk_count"],
        )
        for d in docs
    ]


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete Document",
    description="Remove a document and all its chunks from the index.",
)
async def delete_document(
    document_id: str,
    request: Request,
) -> DeleteResponse:
    """
    Delete a document from the vector index by its ID.
    Also removes the original file from data/documents/.
    """
    pipeline = request.app.state.pipeline

    # Check document exists in index
    docs = pipeline.get_indexed_documents()
    doc_ids = [d["document_id"] for d in docs]

    if document_id not in doc_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    # Remove from FAISS index
    removed = pipeline.delete_document(document_id)

    # Remove file from disk (find by document_id prefix)
    settings = get_settings()
    docs_dir = Path(settings.documents_dir)
    for f in docs_dir.glob(f"{document_id}_*"):
        f.unlink(missing_ok=True)
        logger.info(f"Deleted file: {f}")

    return DeleteResponse(
        document_id=document_id,
        chunks_removed=removed,
        success=True,
        message=f"Successfully removed {removed} chunks from index",
    )


@router.post(
    "/{document_id}/reindex",
    response_model=IngestResponse,
    summary="Re-index Document",
    description="Delete and re-index an existing document.",
)
async def reindex_document(
    document_id: str,
    request: Request,
) -> IngestResponse:
    """
    Re-index an existing document.
    Useful after changing chunk_size or embedding model.
    """
    pipeline = request.app.state.pipeline
    settings = get_settings()

    # Find the file on disk
    docs_dir = Path(settings.documents_dir)
    matching_files = list(docs_dir.glob(f"{document_id}_*"))

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail=f"Document file not found for id: {document_id}",
        )

    file_path = matching_files[0]

    # Extract original filename by removing UUID prefix
    # File is stored as: "{document_id}_{original_filename}"
    stored_name = file_path.name
    original_filename = stored_name[len(document_id) + 1:]  # Remove "uuid_"

    # Delete existing index entries
    pipeline.delete_document(document_id)
    logger.info(f"Deleted existing index for {document_id}")

    # Re-ingest with original filename preserved
    result = pipeline.ingest(
        file_path=file_path,
        document_id=document_id,
        original_filename=original_filename,
    )

    return IngestResponse(
        document_id=result.document_id,
        filename=original_filename,
        chunks_indexed=result.chunks_indexed,
        page_count=result.page_count,
        word_count=result.word_count,
        processing_time=round(result.processing_time, 3),
        status=result.status,
        error=result.error,
    )