"""
pipeline.py — Master RAG Pipeline Orchestrator
===============================================
The RAGPipeline is the single entry point for all operations.
FastAPI routes import ONLY this class.

Responsibilities:
  - Initialize and hold all AI components as singletons
  - ingest(): process and index any supported document
  - ask(): retrieve + generate for any query
  - delete(): remove a document from the index
  - stats(): return system health and index information

Singleton lifecycle:
  Created once at FastAPI startup via lifespan event.
  Shared across all requests via dependency injection.
  Never re-created during the application lifetime.
"""

import time
import uuid
from pathlib import Path
from typing import Optional

from backend.processors.pdf_processor import PDFProcessor
from backend.processors.docx_processor import DOCXProcessor
from backend.processors.chunker import TextChunker
from backend.rag.embeddings import EmbeddingService
from backend.rag.vector_store import FAISSVectorStore
from backend.rag.retriever import Retriever
from backend.rag.llm import OllamaLLM
from backend.models.document_models import (
    IngestResult,
    RAGResponse,
)
from backend.config import get_settings
from backend.utils.logger import logger


class RAGPipeline:
    """
    Master orchestrator for the Local RAG System.

    Initializes all components once and exposes clean methods
    for document ingestion and question answering.

    Args:
        chunk_size:    Override default chunk size from config
        chunk_overlap: Override default chunk overlap from config
        top_k:         Override default retrieval count from config
        llm_model:     Override default Ollama model from config

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.ingest("path/to/document.pdf")
        response = pipeline.ask("What is the main finding?")
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        llm_model: Optional[str] = None,
    ):
        settings = get_settings()
        self.settings = settings

        logger.info("Initializing RAG Pipeline...")
        start = time.time()

        # ── Component 1: Document Processors ──────────────────────
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()

        # ── Component 2: Text Chunker ──────────────────────────────
        self.chunker = TextChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

        # ── Component 3: Embedding Service ────────────────────────
        self.embedding_service = EmbeddingService()

        # ── Component 4: Vector Store ──────────────────────────────
        self.vector_store = FAISSVectorStore(
            index_path=settings.faiss_index_file,
            metadata_path=settings.faiss_metadata_file,
            dimension=settings.embedding_dimension,
        )

        # ── Component 5: Retriever ─────────────────────────────────
        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            chunker=self.chunker,
        )

        # ── Component 6: LLM ──────────────────────────────────────
        self.llm = OllamaLLM(
            model=llm_model or settings.ollama_model,
        )

        # ── Configuration ──────────────────────────────────────────
        self.top_k = top_k or settings.top_k_results
        self.score_threshold = settings.similarity_threshold

        elapsed = time.time() - start
        logger.info(
            f"RAG Pipeline ready | "
            f"Init time: {elapsed:.2f}s | "
            f"Model: {settings.embedding_model} | "
            f"LLM: {self.llm.model} | "
            f"Vectors in index: {self.vector_store.total_vectors}"
        )

    # ── Ingestion ──────────────────────────────────────────────────

    def ingest(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> IngestResult:
        """
        Process and index a document file (PDF or DOCX).

        Args:
            file_path:         Path to the PDF or DOCX file on disk
            document_id:       Unique ID. Auto-generated if not provided.
            original_filename: The real filename to show users.
                               file_path may have a UUID prefix for storage.
                               If not provided, uses file_path.name.

        Returns:
            IngestResult with stats. Never raises.
        """
        start = time.time()
        file_path = Path(file_path)

        if not document_id:
            document_id = str(uuid.uuid4())

        # Use original_filename for display, file_path.name for processing
        display_name = original_filename or file_path.name

        logger.info(
            f"Pipeline ingesting: {display_name} "
            f"(id={document_id})"
        )

        # ── Detect file type and choose processor ──────────────────
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                document = self.pdf_processor.process(file_path)
            elif suffix == ".docx":
                document = self.docx_processor.process(file_path)
            else:
                return IngestResult(
                    document_id=document_id,
                    filename=display_name,
                    chunks_indexed=0,
                    processing_time=time.time() - start,
                    status="failed",
                    error=(
                        f"Unsupported file type: {suffix}. "
                        f"Supported: .pdf, .docx"
                    ),
                )

        except Exception as e:
            logger.error(f"Processing failed for {display_name}: {e}")
            return IngestResult(
                document_id=document_id,
                filename=display_name,
                chunks_indexed=0,
                processing_time=time.time() - start,
                status="failed",
                error=f"File processing error: {str(e)}",
            )

        # ── Check processing result ────────────────────────────────
        if not document.is_successful:
            return IngestResult(
                document_id=document_id,
                filename=display_name,
                chunks_indexed=0,
                processing_time=time.time() - start,
                status="failed",
                error=document.error_message,
                page_count=document.metadata.page_count,
            )

        # ── KEY FIX: Override stored filename with original name ───
        # The file on disk has a UUID prefix (e.g. "abc123_report.pdf")
        # but we want FAISS metadata and UI to show "report.pdf"
        if original_filename:
            document.metadata.filename = original_filename

        # ── Index the document ─────────────────────────────────────
        index_result = self.retriever.index_document(document, document_id)

        elapsed = time.time() - start

        logger.info(
            f"Ingest complete: {display_name} | "
            f"Chunks: {index_result['chunks_indexed']} | "
            f"Time: {elapsed:.2f}s"
        )

        return IngestResult(
            document_id=document_id,
            filename=display_name,
            chunks_indexed=index_result["chunks_indexed"],
            processing_time=elapsed,
            status=index_result["status"],
            error=index_result.get("error"),
            page_count=document.metadata.page_count,
            word_count=document.metadata.total_words,
        )

    # ── Question Answering ─────────────────────────────────────────

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_document_id: Optional[str] = None,
        temperature: float = 0.1,
    ) -> RAGResponse:
        """
        Answer a question using the indexed documents.

        Args:
            query:              The user's question
            top_k:              Number of chunks to retrieve
            score_threshold:    Minimum similarity score
            filter_document_id: Search only within this document
            temperature:        LLM creativity (0.1 = factual)

        Returns:
            RAGResponse with answer, sources, and timing.
            Never raises — errors captured in response.
        """
        total_start = time.time()

        if not query or not query.strip():
            return RAGResponse(
                query=query,
                answer="Please provide a valid question.",
                sources=[],
                model=self.llm.model,
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=0.0,
                chunks_retrieved=0,
                success=False,
                error="Empty query",
            )

        logger.info(f"Pipeline ask: '{query[:60]}'")

        # ── Step 1: Retrieve relevant chunks ──────────────────────
        retrieval_start = time.time()

        try:
            chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k or self.top_k,
                score_threshold=(
                    score_threshold
                    if score_threshold is not None
                    else self.score_threshold
                ),
                filter_document_id=filter_document_id,
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            chunks = []

        retrieval_time = time.time() - retrieval_start

        logger.info(
            f"Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s"
        )

        # ── Step 2: Generate answer with LLM ──────────────────────
        llm_response = self.llm.generate_answer(
            query=query,
            context_chunks=chunks,
            temperature=temperature,
        )

        total_time = time.time() - total_start

        # ── Step 3: Build final response ───────────────────────────
        return RAGResponse(
            query=query,
            answer=llm_response.answer,
            sources=llm_response.unique_sources,
            model=llm_response.model,
            retrieval_time=retrieval_time,
            generation_time=llm_response.generation_time,
            total_time=total_time,
            chunks_retrieved=len(chunks),
            success=llm_response.success,
            error=llm_response.error,
        )

    # ── Document Management ────────────────────────────────────────

    def delete_document(self, document_id: str) -> int:
        """
        Remove a document and all its chunks from the index.

        Args:
            document_id: The document ID returned by ingest()

        Returns:
            Number of chunks removed
        """
        removed = self.retriever.delete_document(document_id)
        logger.info(
            f"Pipeline deleted document {document_id}: "
            f"{removed} chunks removed"
        )
        return removed

    def get_indexed_documents(self) -> list[dict]:
        """Return list of all indexed documents with chunk counts."""
        return self.vector_store.get_indexed_documents()

    # ── Health & Stats ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return full system status.
        Used by the /health and /stats API endpoints.
        """
        return {
            "pipeline_ready": True,
            "llm_available": self.llm.is_available(),
            "llm_model": self.llm.model,
            "available_models": self.llm.list_models(),
            "embedding_model": self.embedding_service.model_name,
            "total_vectors": self.vector_store.total_vectors,
            "index_size_mb": self.vector_store.index_size_mb,
            "indexed_documents": self.vector_store.get_indexed_documents(),
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
        }

    @property
    def is_ready(self) -> bool:
        """True if the pipeline has documents indexed and LLM available."""
        return (
            not self.vector_store.is_empty
            and self.llm.is_available()
        )