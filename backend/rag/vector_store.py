"""
vector_store.py — FAISS Vector Database
=========================================
Manages the FAISS index and associated metadata.

Persistence model:
  - data/vector_store/index.faiss    ← FAISS binary index
  - data/vector_store/metadata.json  ← Parallel metadata list

The two files are always written together atomically to prevent
corruption from partial writes.

Thread safety:
  This implementation is not thread-safe for concurrent writes.
  For production multi-user systems, add a write lock.
  For Phase 1, single-user sequential writes are fine.
"""

import json
import time
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from backend.models.document_models import DocumentChunk, SearchResult
from backend.config import get_settings
from backend.utils.logger import logger


class FAISSVectorStore:
    """
    Persistent vector store backed by FAISS IndexFlatIP.

    IndexFlatIP:
      - "Flat"  = brute-force exact search (no approximation)
      - "IP"    = Inner Product similarity metric
      - Because our vectors are L2-normalized, IP == cosine similarity
      - Best choice for < 100K vectors: exact, simple, fast

    Args:
        index_path:    Path to save/load the FAISS binary index
        metadata_path: Path to save/load the JSON metadata
        dimension:     Embedding vector size (must match model)

    Usage:
        store = FAISSVectorStore()
        store.add_chunks(chunks, vectors)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        settings = get_settings()

        self.index_path = Path(index_path or settings.faiss_index_file)
        self.metadata_path = Path(
            metadata_path or settings.faiss_metadata_file
        )
        self.dimension = dimension or settings.embedding_dimension

        # ── In-memory state ────────────────────────────────────────
        # These two lists are ALWAYS kept in sync.
        # metadata[i] describes the vector at index position i.
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: list[dict] = []

        # ── Initialize ─────────────────────────────────────────────
        self._initialize()

    def _initialize(self) -> None:
        """
        Load existing index from disk, or create a fresh one.

        Called once at construction. After this, the store is ready.
        """
        if self.index_path.exists() and self.metadata_path.exists():
            self._load()
        else:
            logger.info("No existing FAISS index found — creating new one")
            self._create_new_index()

    def _create_new_index(self) -> None:
        """
        Create a new empty FAISS index.

        IndexFlatIP is the simplest FAISS index:
          - Stores vectors as-is (no compression)
          - Searches by inner product (= cosine sim for unit vectors)
          - Exact search — never misses a match
          - Memory: dimension × 4 bytes per vector
            (384 × 4 = 1536 bytes = 1.5KB per chunk)
            10,000 chunks = ~15MB — very manageable
        """
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = []
        logger.info(
            f"Created new FAISS IndexFlatIP | dimension={self.dimension}"
        )

    # ── Core Operations ────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: np.ndarray,
    ) -> int:
        """
        Add document chunks and their embeddings to the store.

        Args:
            chunks:  DocumentChunk list (provides metadata)
            vectors: numpy array shape (len(chunks), dimension)
                     Must be float32 and L2-normalized

        Returns:
            Number of chunks successfully added

        Raises:
            ValueError: If chunks and vectors count mismatch
            ValueError: If vectors have wrong dimension
        """
        # ── Validation ─────────────────────────────────────────────
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunks count ({len(chunks)}) != "
                f"vectors count ({len(vectors)})"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != "
                f"expected {self.dimension}"
            )

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        logger.info(
            f"Adding {len(chunks)} chunks to FAISS index | "
            f"Current size: {self.total_vectors}"
        )

        start = time.time()

        # ── Add vectors to FAISS ───────────────────────────────────
        # index.add() appends vectors to the index.
        # The position in the index (0, 1, 2, ...) is implicit —
        # it's just the order they were added.
        self._index.add(vectors)

        # ── Add parallel metadata ──────────────────────────────────
        # For each chunk, store everything we need to reconstruct
        # a SearchResult without going back to the original document.
        for chunk in chunks:
            self._metadata.append({
                "chunk_id":      chunk.chunk_id,
                "text":          chunk.text,
                "document_id":   chunk.metadata.document_id,
                "filename":      chunk.metadata.filename,
                "document_type": chunk.metadata.document_type,
                "page_number":   chunk.metadata.page_number,
                "chunk_index":   chunk.metadata.chunk_index,
                "total_chunks":  chunk.metadata.total_chunks,
                "char_start":    chunk.metadata.char_start,
                "char_end":      chunk.metadata.char_end,
            })

        # ── Persist to disk ────────────────────────────────────────
        self.save()

        elapsed = time.time() - start
        logger.info(
            f"Added {len(chunks)} chunks | "
            f"Total in index: {self.total_vectors} | "
            f"Time: {elapsed:.2f}s"
        )

        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_document_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar chunks to a query vector.

        Args:
            query_vector:       1D numpy array, shape (dimension,)
            top_k:              Number of results to return
            score_threshold:    Minimum similarity score (0.0-1.0)
                                Chunks below this are filtered out
            filter_document_id: If set, only return results from
                                this specific document

        Returns:
            List of SearchResult, sorted by score descending.
            May be shorter than top_k if threshold filters results.

        Note:
            Returns empty list (not exception) when index is empty.
        """
        if self.total_vectors == 0:
            logger.warning("Search called on empty index")
            return []

        # ── Reshape query vector ───────────────────────────────────
        # FAISS search() requires shape (n_queries, dimension)
        # We have 1 query, so reshape (384,) → (1, 384)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # ── Determine how many to fetch ────────────────────────────
        # If filtering by document, we might need more raw results
        # to end up with top_k after filtering
        fetch_k = min(
            top_k * 3 if filter_document_id else top_k,
            self.total_vectors,
        )

        # ── FAISS search ───────────────────────────────────────────
        # Returns two arrays:
        #   distances: shape (1, fetch_k) — similarity scores
        #   indices:   shape (1, fetch_k) — positions in index
        # Both sorted by distance descending (highest sim first)
        distances, indices = self._index.search(query_vector, fetch_k)

        # Flatten from (1, k) to (k,)
        distances = distances[0]
        indices = indices[0]

        # ── Build SearchResult objects ─────────────────────────────
        results = []

        for score, idx in zip(distances, indices):
            # FAISS returns -1 for "not found" slots (when k > index size)
            if idx == -1:
                continue

            # Apply score threshold
            if score < score_threshold:
                continue

            # Get metadata for this vector position
            meta = self._metadata[idx]

            # Apply document filter if requested
            if (filter_document_id and
                    meta["document_id"] != filter_document_id):
                continue

            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=float(score),
                filename=meta["filename"],
                document_id=meta["document_id"],
                page_number=meta["page_number"],
                chunk_index=meta["chunk_index"],
                document_type=meta["document_type"],
            ))

            # Stop when we have enough results
            if len(results) >= top_k:
                break

        logger.debug(
            f"Search complete | "
            f"top_k={top_k} | "
            f"found={len(results)} | "
            f"threshold={score_threshold}"
        )

        return results

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document from the index.

        FAISS IndexFlatIP doesn't support in-place deletion.
        Strategy: rebuild the index without the deleted document's vectors.

        Args:
            document_id: The document ID to remove

        Returns:
            Number of chunks removed

        Note:
            This is O(N) — rebuilds entire index.
            Acceptable for small-medium collections.
            For large collections, use IndexIDMap instead (Phase 17).
        """
        if self.total_vectors == 0:
            logger.warning(f"Delete called on empty index")
            return 0

        # ── Find positions to keep ─────────────────────────────────
        keep_indices = [
            i for i, meta in enumerate(self._metadata)
            if meta["document_id"] != document_id
        ]
        removed_count = self.total_vectors - len(keep_indices)

        if removed_count == 0:
            logger.warning(f"Document not found in index: {document_id}")
            return 0

        logger.info(
            f"Deleting document {document_id} | "
            f"Removing {removed_count} chunks | "
            f"Keeping {len(keep_indices)} chunks"
        )

        # ── Extract all current vectors from FAISS ─────────────────
        # reconstruct_n(start, end) extracts raw vectors from the index
        all_vectors = np.zeros(
            (self.total_vectors, self.dimension), dtype=np.float32
        )
        self._index.reconstruct_n(0, self.total_vectors, all_vectors)

        # ── Keep only non-deleted vectors and metadata ─────────────
        kept_vectors = all_vectors[keep_indices]
        kept_metadata = [self._metadata[i] for i in keep_indices]

        # ── Rebuild fresh index ────────────────────────────────────
        self._create_new_index()
        self._metadata = kept_metadata

        if len(kept_vectors) > 0:
            self._index.add(kept_vectors)

        # ── Save ───────────────────────────────────────────────────
        self.save()

        logger.info(
            f"Document deleted: {document_id} | "
            f"Index now has {self.total_vectors} vectors"
        )

        return removed_count

    def get_document_chunk_count(self, document_id: str) -> int:
        """Return how many chunks a document has in the index."""
        return sum(
            1 for meta in self._metadata
            if meta["document_id"] == document_id
        )

    def get_indexed_documents(self) -> list[dict]:
        """
        Return a summary of all documents in the index.
        Useful for the /documents API endpoint.
        """
        seen = {}
        for meta in self._metadata:
            doc_id = meta["document_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "document_type": meta["document_type"],
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1

        return list(seen.values())

    # ── Persistence ────────────────────────────────────────────────

    def save(self) -> None:
        """
        Save FAISS index and metadata to disk.

        Write order: metadata first, then index.
        If the process crashes mid-write, we'd rather have
        stale metadata than an index with no metadata.

        Uses atomic write pattern: write to temp file, then rename.
        This prevents partial/corrupt writes on crash.
        """
        try:
            # ── Ensure directories exist ───────────────────────────
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # ── Save metadata (atomic write via temp file) ─────────
            temp_meta = self.metadata_path.with_suffix(".tmp")
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
            temp_meta.replace(self.metadata_path)  # Atomic rename

            # ── Save FAISS index ───────────────────────────────────
            faiss.write_index(self._index, str(self.index_path))

            logger.debug(
                f"Saved index: {self.total_vectors} vectors | "
                f"{self.index_path}"
            )

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def _load(self) -> None:
        """
        Load FAISS index and metadata from disk.

        Validates that vector count matches metadata count.
        If mismatch detected, resets to empty (safe recovery).
        """
        try:
            logger.info(f"Loading FAISS index from: {self.index_path}")

            # Load FAISS index
            self._index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            # Validate consistency
            if self._index.ntotal != len(self._metadata):
                logger.error(
                    f"Index/metadata mismatch: "
                    f"{self._index.ntotal} vectors vs "
                    f"{len(self._metadata)} metadata entries. "
                    f"Resetting to empty index."
                )
                self._create_new_index()
                return

            logger.info(
                f"Loaded FAISS index: {self.total_vectors} vectors | "
                f"dimension={self.dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}. Starting fresh.")
            self._create_new_index()

    def reset(self) -> None:
        """
        Clear the entire index and delete files from disk.
        Used in testing. Use with caution in production.
        """
        self._create_new_index()

        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

        logger.warning("FAISS index reset and files deleted")

    # ── Properties ─────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal if self._index else 0

    @property
    def is_empty(self) -> bool:
        return self.total_vectors == 0

    @property
    def index_size_mb(self) -> float:
        """Approximate memory size of the index in MB."""
        bytes_per_vector = self.dimension * 4  # float32 = 4 bytes
        return (self.total_vectors * bytes_per_vector) / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore("
            f"vectors={self.total_vectors}, "
            f"dimension={self.dimension}, "
            f"size={self.index_size_mb:.2f}MB)"
        )