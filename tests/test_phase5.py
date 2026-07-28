"""
test_phase5.py — Embedding Service Verification
================================================
NOTE: First run will download the model (~90MB).
      Subsequent runs load from cache in ~2 seconds.

Run with:
    python tests/test_phase5.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.embeddings import EmbeddingService
from backend.utils.logger import setup_logger

setup_logger(debug=False)

# ── Shared service instance (loaded once for all tests) ────────────
print("\nLoading embedding model (first run downloads ~90MB)...")
embedding_service = EmbeddingService()


def test_model_loads():
    """Verify model info is correct after loading."""
    print("\n── Test 1: Model Loaded ──────────────────────────────")

    info = embedding_service.model_info
    print(f"   Model:     {info['model_name']}")
    print(f"   Dimension: {info['dimension']}")
    print(f"   Loaded:    {info['model_loaded']}")

    assert info["model_loaded"] is True
    assert info["dimension"] == 384
    print("   ✅ Model loaded successfully")


def test_query_embedding_shape():
    """Verify query embeddings have correct shape and dtype."""
    print("\n── Test 2: Query Embedding Shape ────────────────────")

    query = "What is retrieval augmented generation?"
    vector = embedding_service.embed_query(query)

    print(f"   Query: '{query}'")
    print(f"   Shape: {vector.shape}")
    print(f"   Dtype: {vector.dtype}")
    print(f"   Min:   {vector.min():.4f}")
    print(f"   Max:   {vector.max():.4f}")
    print(f"   Norm:  {np.linalg.norm(vector):.4f}  (should be ~1.0)")

    assert vector.shape == (384,), f"Expected (384,), got {vector.shape}"
    assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"
    # Normalized vectors have L2 norm = 1.0 (within floating point tolerance)
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5, "Vector not normalized!"
    print("   ✅ Query embedding shape and dtype correct")


def test_semantic_similarity():
    """
    The most important test: semantically similar texts
    should have HIGH similarity, unrelated texts LOW similarity.
    """
    print("\n── Test 3: Semantic Similarity ──────────────────────")

    pairs = [
        # (text_a, text_b, expected_relationship)
        (
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence.",
            "HIGH",
        ),
        (
            "How do neural networks work?",
            "Deep learning uses layers of neurons to process data.",
            "HIGH",
        ),
        (
            "What is the capital of France?",
            "FAISS is a vector similarity search library.",
            "LOW",
        ),
        (
            "Retrieval Augmented Generation improves LLM accuracy.",
            "RAG combines document retrieval with language generation.",
            "HIGH",
        ),
    ]

    print(f"   {'Text A':<45} {'Text B':<45} {'Score':>6} {'Expected'}")
    print(f"   {'-'*45} {'-'*45} {'-'*6} {'-'*8}")

    for text_a, text_b, expected in pairs:
        vec_a = embedding_service.embed_query(text_a)
        vec_b = embedding_service.embed_query(text_b)
        score = embedding_service.compute_similarity(vec_a, vec_b)

        status = "✅" if (
            (expected == "HIGH" and score > 0.4) or
            (expected == "LOW" and score < 0.4)
        ) else "❌"

        a_short = text_a[:42] + "..." if len(text_a) > 45 else text_a
        b_short = text_b[:42] + "..." if len(text_b) > 45 else text_b
        print(f"   {a_short:<45} {b_short:<45} {score:>6.3f} {expected} {status}")

    print("\n   ✅ Semantic similarity working correctly")


def test_batch_embedding():
    """Test embedding multiple chunks at once."""
    print("\n── Test 4: Batch Chunk Embedding ────────────────────")

    from backend.models.document_models import (
        DocumentChunk, ChunkMetadata, DocumentType
    )

    # Create 10 test chunks
    test_texts = [
        "Python is a high-level programming language.",
        "FastAPI is a modern web framework for Python.",
        "FAISS enables efficient vector similarity search.",
        "Sentence Transformers generate semantic embeddings.",
        "RAG combines retrieval with language generation.",
        "SQLite is a lightweight embedded database.",
        "React is a JavaScript library for building UIs.",
        "Docker containers package applications consistently.",
        "Ollama runs large language models locally.",
        "Jaipur is the Pink City of Rajasthan, India.",
    ]

    chunks = [
        DocumentChunk(
            chunk_id=f"test_chunk_{i:04d}",
            text=text,
            metadata=ChunkMetadata(
                document_id="test_doc",
                filename="test.pdf",
                document_type="pdf",
                page_number=1,
                chunk_index=i,
                total_chunks=len(test_texts),
                char_start=0,
                char_end=len(text),
            ),
        )
        for i, text in enumerate(test_texts)
    ]

    start = time.time()
    vectors = embedding_service.embed_chunks(chunks, show_progress=False)
    elapsed = time.time() - start

    print(f"   Chunks embedded: {len(chunks)}")
    print(f"   Output shape:    {vectors.shape}")
    print(f"   Output dtype:    {vectors.dtype}")
    print(f"   Time taken:      {elapsed:.3f}s")
    print(f"   Speed:           {len(chunks)/elapsed:.0f} chunks/sec")

    assert vectors.shape == (10, 384), f"Expected (10, 384), got {vectors.shape}"
    assert vectors.dtype == np.float32

    # All vectors should be normalized (L2 norm ≈ 1.0)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Vectors not normalized: {norms}"
    print(f"   All norms ≈ 1.0: {norms.round(4).tolist()}")

    print(f"   ✅ Batch embedding passed")


def test_different_queries_different_vectors():
    """Verify that different queries produce different vectors."""
    print("\n── Test 5: Embedding Uniqueness ─────────────────────")

    queries = [
        "What is machine learning?",
        "How do I cook pasta?",
        "Explain quantum entanglement.",
    ]

    vectors = [embedding_service.embed_query(q) for q in queries]

    # Check that all pairs are different
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = embedding_service.compute_similarity(vectors[i], vectors[j])
            print(f"   Similarity({i},{j}): {sim:.4f}  ({queries[i][:30]!r} vs {queries[j][:30]!r})")
            assert sim < 0.99, "Two different queries produced identical vectors!"

    print("   ✅ Different queries → different vectors confirmed")


def test_real_pdf_embedding():
    """End-to-end: PDF → chunks → embeddings."""
    print("\n── Test 6: Real PDF End-to-End ──────────────────────")

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  sample.pdf not found — skipping")
        return

    from backend.processors.pdf_processor import PDFProcessor
    from backend.processors.chunker import TextChunker

    # Process PDF
    processor = PDFProcessor()
    doc = processor.process(pdf_path)

    # Chunk it
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(doc, document_id="sample_pdf_e2e")

    print(f"   Chunks to embed: {len(chunks)}")

    # Embed all chunks
    start = time.time()
    vectors = embedding_service.embed_chunks(chunks, show_progress=True)
    elapsed = time.time() - start

    print(f"   Embedding shape: {vectors.shape}")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Speed: {len(chunks)/elapsed:.0f} chunks/sec")

    assert vectors.shape[0] == len(chunks)
    assert vectors.shape[1] == 384

    # Test semantic search manually:
    # Find which chunk is most similar to a query
    query = "Netflix content recommendation system"
    query_vec = embedding_service.embed_query(query)

    # Compute cosine similarities (dot product of normalized vectors)
    similarities = np.dot(vectors, query_vec)

    # Get top 3 most similar chunks
    top_indices = np.argsort(similarities)[::-1][:3]

    print(f"\n   Query: '{query}'")
    print(f"   Top 3 most similar chunks:")
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        chunk = chunks[idx]
        print(f"\n   Rank {rank} | Score: {score:.4f} | "
              f"Page {chunk.metadata.page_number}")
        print(f"   '{chunk.text[:100]}...'")

    assert similarities[top_indices[0]] > 0.3, \
        "Best match should have similarity > 0.3"
    print(f"\n   ✅ Real PDF embedding and search working")


def test_empty_query_raises():
    """Verify that empty queries raise ValueError."""
    print("\n── Test 7: Input Validation ─────────────────────────")

    try:
        embedding_service.embed_query("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✅ Empty query rejected: {e}")

    try:
        embedding_service.embed_query("   ")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✅ Whitespace-only query rejected: {e}")

    try:
        embedding_service.embed_chunks([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✅ Empty chunks list rejected: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 5: EMBEDDING TESTS")
    print("=" * 55)

    test_model_loads()
    test_query_embedding_shape()
    test_semantic_similarity()
    test_batch_embedding()
    test_different_queries_different_vectors()
    test_real_pdf_embedding()
    test_empty_query_raises()

    print("\n" + "=" * 55)
    print("   Phase 5 complete!")
    print("   Next: confirm all ✅ → Phase 6 (FAISS Vector Store)")
    print("=" * 55 + "\n")