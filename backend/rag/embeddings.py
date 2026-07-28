"""
embeddings.py — Local Embedding Service
========================================
Converts text into dense vector representations using
Sentence Transformers running entirely on local hardware.

Key design decisions:
  1. Singleton model: loaded once at startup, reused forever.
     Loading a transformer model takes ~2 seconds. We cannot
     do this per request.

  2. Batch processing: embedding 50 chunks at once is faster
     than embedding them one at a time. The GPU/CPU can
     parallelize matrix operations across the batch.

  3. Normalized vectors: we use normalize_embeddings=True.
     This makes cosine similarity computable as a simple
     dot product — which is what FAISS's IndexFlatIP uses.

  4. float32 precision: FAISS requires float32. We explicitly
     cast to ensure consistency regardless of model output type.
"""

import time
import numpy as np
from typing import Optional

# SentenceTransformer is the main class from the sentence-transformers library
# It wraps a HuggingFace transformer model with convenient encode() method
from sentence_transformers import SentenceTransformer

from backend.models.document_models import DocumentChunk
from backend.config import get_settings
from backend.utils.logger import logger


class EmbeddingService:
    """
    Manages local text embeddings using Sentence Transformers.

    Designed as a singleton: instantiate once and inject
    everywhere it's needed. The model stays loaded in memory.

    Args:
        model_name: HuggingFace model name or local path.
                    Defaults to config value (all-MiniLM-L6-v2).
        batch_size: How many chunks to embed per GPU/CPU batch.
                    Larger = faster but more memory. 32 is safe.

    Usage:
        service = EmbeddingService()
        vectors = service.embed_chunks(my_chunks)     # (N, 384)
        query_vec = service.embed_query("my question") # (384,)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None

        # Load the model immediately on construction
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the Sentence Transformer model into memory.

        First run: downloads model from HuggingFace (~90MB for MiniLM).
        Subsequent runs: loads from local cache (~2 seconds).
        Cache location: ~/.cache/huggingface/hub/

        This is called once in __init__ and never again.
        """
        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(
            "First run will download the model (~90MB). "
            "Subsequent runs load from cache."
        )

        start = time.time()

        try:
            # SentenceTransformer automatically handles:
            # - Downloading from HuggingFace if not cached
            # - Loading from local cache if available
            # - Moving to GPU if available (falls back to CPU)
            self._model = SentenceTransformer(self.model_name)

            load_time = time.time() - start
            logger.info(
                f"Model loaded successfully: {self.model_name} | "
                f"Time: {load_time:.2f}s | "
                f"Dimension: {self.dimension}"
            )

        except Exception as e:
            # Model loading failure is fatal — we cannot embed without a model
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Could not load embedding model '{self.model_name}'. "
                f"Check your internet connection for first-time download. "
                f"Error: {e}"
            )

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of DocumentChunks.

        Processes in batches for efficiency. Returns a 2D numpy
        array where row i is the embedding for chunks[i].

        Args:
            chunks:        List of DocumentChunk from chunker.py
            show_progress: Show tqdm progress bar during encoding

        Returns:
            numpy array of shape (len(chunks), embedding_dimension)
            dtype: float32

        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If model isn't loaded
        """
        if not chunks:
            raise ValueError("Cannot embed empty chunks list")

        if self._model is None:
            raise RuntimeError("Embedding model not loaded")

        logger.info(
            f"Embedding {len(chunks)} chunks | "
            f"Batch size: {self.batch_size}"
        )

        start = time.time()

        # Extract just the text from each chunk
        # The model only needs the string content
        texts = [chunk.text for chunk in chunks]

        # encode() is the main method:
        #   - batch_size: process this many at once (memory vs speed tradeoff)
        #   - show_progress_bar: tqdm bar in terminal
        #   - convert_to_numpy: return np.ndarray (vs torch.Tensor)
        #   - normalize_embeddings: L2 normalize each vector to unit length
        #     This is CRITICAL for cosine similarity to work correctly
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # Unit vectors for cosine sim
        )

        # Ensure float32 — FAISS requirement
        # Some models output float16 or float64; FAISS only accepts float32
        embeddings = embeddings.astype(np.float32)

        elapsed = time.time() - start
        chunks_per_sec = len(chunks) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Embedding complete: {len(chunks)} chunks | "
            f"Shape: {embeddings.shape} | "
            f"Time: {elapsed:.2f}s | "
            f"Speed: {chunks_per_sec:.0f} chunks/sec"
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate a single embedding for a search query.

        This is used at query time (not indexing time).
        The returned vector is compared against all stored
        chunk vectors in FAISS to find the most similar ones.

        Args:
            query: The user's question or search text

        Returns:
            1D numpy array of shape (embedding_dimension,)
            dtype: float32

        Note:
            We use the same model and normalization as embed_chunks()
            so vectors are in the same space and comparable.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self._model is None:
            raise RuntimeError("Embedding model not loaded")

        logger.debug(f"Embedding query: '{query[:60]}...'")

        # encode() with a single string returns a 1D array
        # We wrap in normalize for consistency
        embedding = self._model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw strings (not DocumentChunks).

        Convenience method for cases where you have plain text
        rather than DocumentChunk objects. Used in testing.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            raise ValueError("Cannot embed empty texts list")

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def compute_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Because our vectors are L2-normalized (unit vectors),
        cosine similarity = dot product. This is a fast O(d) op.

        Args:
            vec_a: First vector (1D, shape: (d,))
            vec_b: Second vector (1D, shape: (d,))

        Returns:
            Similarity score in range [-1.0, 1.0]
            1.0  = identical meaning
            0.0  = unrelated
           -1.0  = opposite meaning (rare in practice)
        """
        # np.dot on two unit vectors = cosine similarity
        return float(np.dot(vec_a, vec_b))

    @property
    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "model_loaded": self._model is not None,
        }