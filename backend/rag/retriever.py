"""
retriever.py — RAG Retrieval Orchestrator
==========================================
The Retriever is the single entry point for all retrieval operations.
It coordinates the EmbeddingService, FAISSVectorStore, and TextChunker
into two clean operations:

  index_document()  — process and store a document
  retrieve()        — find relevant chunks for a query

Design pattern: Facade
  Callers (API routes, services) import only Retriever.
  They never touch FAISS, embeddings, or chunking directly.
  This decouples the API layer from the AI layer completely.
"""

import time
from typing import Optional

from backend.rag.embeddings import EmbeddingService
from backend.rag.vector_store import FAISSVectorStore
from backend.processors.chunker import TextChunker
from backend.models.document_models import (
    ProcessedDocument,
    SearchResult,
    DocumentChunk,
)
from backend.config import get_settings
from backend.utils.logger import logger


class Retriever:
    """
    Orchestrates document indexing and similarity retrieval.

    Holds references to all AI components as injected dependencies.
    This makes it easy to swap components or mock them in tests.

    Args:
        embedding_service: Loaded EmbeddingService instance
        vector_store:      Initialized FAISSVectorStore instance
        chunker:           Configured TextChunker instance

    Usage:
        retriever = Retriever(embedding_service, vector_store, chunker)

        # Index a document
        result = retriever.index_document(processed_doc, "doc_id_123")

        # Retrieve relevant chunks
        chunks = retriever.retrieve("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        chunker: Optional[TextChunker] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()
        self.settings = get_settings()

        logger.info("Retriever initialized")

    # ── Indexing ───────────────────────────────────────────────────

    def index_document(
        self,
        document: ProcessedDocument,
        document_id: str,
    ) -> dict:
        """
        Process and index a document into the vector store.

        Full pipeline:
          ProcessedDocument → chunks → embeddings → FAISS

        Args:
            document:    Output from PDFProcessor or DOCXProcessor
            document_id: Unique ID for this document

        Returns:
            Dict with indexing statistics:
            {
                "document_id": str,
                "filename": str,
                "chunks_indexed": int,
                "status": "success" | "failed" | "empty",
                "processing_time": float,
                "error": Optional[str],
            }
        """
        start = time.time()

        logger.info(
            f"Indexing document: {document.metadata.filename} "
            f"(id={document_id})"
        )

        # ── Guard: empty document ──────────────────────────────────
        if not document.is_successful or not document.full_text.strip():
            logger.warning(
                f"Skipping indexing — document not successful: "
                f"{document.metadata.filename} ({document.status})"
            )
            return {
                "document_id": document_id,
                "filename": document.metadata.filename,
                "chunks_indexed": 0,
                "status": "empty",
                "processing_time": time.time() - start,
                "error": document.error_message,
            }

        try:
            # ── Step 1: Chunk ──────────────────────────────────────
            chunks = self.chunker.chunk_document(document, document_id)

            if not chunks:
                logger.warning(
                    f"No chunks produced for {document.metadata.filename}"
                )
                return {
                    "document_id": document_id,
                    "filename": document.metadata.filename,
                    "chunks_indexed": 0,
                    "status": "empty",
                    "processing_time": time.time() - start,
                    "error": "Document produced no indexable chunks",
                }

            # ── Step 2: Embed ──────────────────────────────────────
            vectors = self.embedding_service.embed_chunks(
                chunks, show_progress=False
            )

            # ── Step 3: Store ──────────────────────────────────────
            self.vector_store.add_chunks(chunks, vectors)

            elapsed = time.time() - start

            logger.info(
                f"Indexed: {document.metadata.filename} | "
                f"Chunks: {len(chunks)} | "
                f"Time: {elapsed:.2f}s"
            )

            return {
                "document_id": document_id,
                "filename": document.metadata.filename,
                "chunks_indexed": len(chunks),
                "status": "success",
                "processing_time": elapsed,
                "error": None,
            }

        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"Indexing failed for {document.metadata.filename}: {e}"
            )
            return {
                "document_id": document_id,
                "filename": document.metadata.filename,
                "chunks_indexed": 0,
                "status": "failed",
                "processing_time": elapsed,
                "error": str(e),
            }

    def delete_document(self, document_id: str) -> int:
        """
        Remove a document's chunks from the vector store.

        Args:
            document_id: The document to remove

        Returns:
            Number of chunks removed
        """
        removed = self.vector_store.delete_document(document_id)
        logger.info(
            f"Deleted document {document_id}: {removed} chunks removed"
        )
        return removed

    # ── Retrieval ──────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_document_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Find the most relevant chunks for a query.

        Args:
            query:              The user's question or search text
            top_k:              Number of results (default from config)
            score_threshold:    Minimum similarity (default from config)
            filter_document_id: Limit search to one document

        Returns:
            List of SearchResult, sorted by relevance (highest first).
            Empty list if nothing meets the threshold or index is empty.

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        top_k = top_k or self.settings.top_k_results
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.settings.similarity_threshold
        )

        logger.info(
            f"Retrieving: '{query[:60]}' | "
            f"top_k={top_k} | threshold={score_threshold}"
        )

        start = time.time()

        # ── Step 1: Embed the query ────────────────────────────────
        query_vector = self.embedding_service.embed_query(query)

        # ── Step 2: Search FAISS ───────────────────────────────────
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_document_id=filter_document_id,
        )

        # ── Step 3: Deduplicate ────────────────────────────────────
        # Occasionally the same chunk appears twice (overlap artifact)
        results = self._deduplicate(results)

        elapsed = time.time() - start

        logger.info(
            f"Retrieved {len(results)} results in {elapsed:.3f}s | "
            f"Top score: {results[0].score:.4f}" if results
            else f"Retrieved 0 results in {elapsed:.3f}s"
        )

        return results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove duplicate chunks from results.

        Deduplication is by chunk_id — if the same chunk appears
        twice (shouldn't happen with IndexFlatIP but can with
        some FAISS index types), keep only the highest-scored copy.
        """
        seen_ids = set()
        deduped = []

        for result in results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                deduped.append(result)

        return deduped

    # ── Stats / Info ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return current state of the retrieval system.
        Used by the /health and /stats API endpoints.
        """
        return {
            "total_vectors": self.vector_store.total_vectors,
            "index_size_mb": round(self.vector_store.index_size_mb, 3),
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension,
            "indexed_documents": self.vector_store.get_indexed_documents(),
            "is_ready": not self.vector_store.is_empty,
        }