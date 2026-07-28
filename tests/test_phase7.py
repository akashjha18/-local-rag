"""
test_phase7.py — Retriever Verification
========================================
Run with:
    python tests/test_phase7.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.embeddings import EmbeddingService
from backend.rag.vector_store import FAISSVectorStore
from backend.rag.retriever import Retriever
from backend.processors.chunker import TextChunker
from backend.utils.logger import setup_logger

setup_logger(debug=False)

TEST_INDEX  = "data/vector_store/test_retriever.faiss"
TEST_META   = "data/vector_store/test_retriever_metadata.json"


def make_retriever() -> Retriever:
    """Build a fresh retriever with a clean test store."""
    emb = EmbeddingService()
    store = FAISSVectorStore(
        index_path=TEST_INDEX,
        metadata_path=TEST_META,
        dimension=384,
    )
    store.reset()
    chunker = TextChunker(chunk_size=300, chunk_overlap=30)
    return Retriever(emb, store, chunker)


def index_sample_pdf(retriever: Retriever) -> int:
    """Helper: index the sample PDF and return chunk count."""
    from backend.processors.pdf_processor import PDFProcessor

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        return 0

    doc = PDFProcessor().process(pdf_path)
    result = retriever.index_document(doc, document_id="sample_pdf")
    return result["chunks_indexed"]


print("\nInitializing retriever...")
_shared_emb = EmbeddingService()


def make_retriever_shared() -> Retriever:
    """Reuse the loaded model across tests — faster."""
    store = FAISSVectorStore(
        index_path=TEST_INDEX,
        metadata_path=TEST_META,
        dimension=384,
    )
    store.reset()
    return Retriever(_shared_emb, store, TextChunker(chunk_size=300, chunk_overlap=30))


def test_index_document():
    """Test that index_document returns correct stats."""
    print("\n── Test 1: Index Document ───────────────────────────")

    retriever = make_retriever_shared()
    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  sample.pdf not found")
        return

    from backend.processors.pdf_processor import PDFProcessor
    doc = PDFProcessor().process(pdf_path)

    result = retriever.index_document(doc, "test_index_doc")

    print(f"   Status:        {result['status']}")
    print(f"   Chunks indexed:{result['chunks_indexed']}")
    print(f"   Filename:      {result['filename']}")
    print(f"   Time:          {result['processing_time']:.2f}s")
    print(f"   Error:         {result['error']}")

    assert result["status"] == "success"
    assert result["chunks_indexed"] > 0
    assert result["error"] is None
    print("   ✅ index_document working")


def test_retrieve_basic():
    """Test basic retrieval returns ranked results."""
    print("\n── Test 2: Basic Retrieval ──────────────────────────")

    retriever = make_retriever_shared()
    n = index_sample_pdf(retriever)
    if n == 0:
        print("   ⏭️  No document to search")
        return

    results = retriever.retrieve("recommendation system algorithm", top_k=3)

    print(f"   Query: 'recommendation system algorithm'")
    print(f"   Results: {len(results)}")
    for r in results:
        print(f"\n   [{r.confidence_percent}%] {r.source_label}")
        print(f"   '{r.text[:80]}...'")

    assert len(results) > 0
    assert results[0].score >= results[-1].score  # Descending order
    print("\n   ✅ Basic retrieval working")


def test_retrieve_threshold():
    """Test that threshold filters irrelevant results."""
    print("\n── Test 3: Retrieval Threshold ──────────────────────")

    retriever = make_retriever_shared()
    n = index_sample_pdf(retriever)
    if n == 0:
        print("   ⏭️  No document to search")
        return

    # Query completely unrelated to the document
    results_low  = retriever.retrieve(
        "ancient roman architecture and aqueducts",
        top_k=5, score_threshold=0.0
    )
    results_high = retriever.retrieve(
        "ancient roman architecture and aqueducts",
        top_k=5, score_threshold=0.5
    )

    print(f"   Irrelevant query results (threshold=0.0): {len(results_low)}")
    if results_low:
        print(f"   Best score: {results_low[0].score:.4f}")
    print(f"   Irrelevant query results (threshold=0.5): {len(results_high)}")

    assert len(results_high) <= len(results_low)
    print("   ✅ Threshold filtering working")


def test_retrieve_empty_query_raises():
    """Test that empty queries raise ValueError."""
    print("\n── Test 4: Empty Query Validation ───────────────────")

    retriever = make_retriever_shared()

    try:
        retriever.retrieve("")
        assert False, "Should have raised"
    except ValueError as e:
        print(f"   ✅ Empty query rejected: {e}")

    try:
        retriever.retrieve("   ")
        assert False, "Should have raised"
    except ValueError as e:
        print(f"   ✅ Whitespace query rejected: {e}")


def test_delete_via_retriever():
    """Test document deletion through the Retriever interface."""
    print("\n── Test 5: Delete Document ──────────────────────────")

    retriever = make_retriever_shared()
    n = index_sample_pdf(retriever)
    if n == 0:
        print("   ⏭️  No document indexed")
        return

    stats_before = retriever.get_stats()
    print(f"   Vectors before deletion: {stats_before['total_vectors']}")

    removed = retriever.delete_document("sample_pdf")
    stats_after = retriever.get_stats()

    print(f"   Chunks removed: {removed}")
    print(f"   Vectors after deletion: {stats_after['total_vectors']}")

    assert stats_after["total_vectors"] == 0
    assert removed == n
    print("   ✅ Delete via Retriever working")


def test_get_stats():
    """Test the stats method returns correct info."""
    print("\n── Test 6: Retriever Stats ──────────────────────────")

    retriever = make_retriever_shared()
    n = index_sample_pdf(retriever)

    stats = retriever.get_stats()

    print(f"   total_vectors:    {stats['total_vectors']}")
    print(f"   index_size_mb:    {stats['index_size_mb']}")
    print(f"   embedding_model:  {stats['embedding_model']}")
    print(f"   is_ready:         {stats['is_ready']}")
    print(f"   indexed_docs:     {stats['indexed_documents']}")

    assert stats["total_vectors"] == n
    assert stats["embedding_model"] == "all-MiniLM-L6-v2"
    assert stats["is_ready"] is True
    assert len(stats["indexed_documents"]) == 1
    print("   ✅ Stats correct")


def test_empty_index_retrieve():
    """Test retrieval on empty index returns empty list, not error."""
    print("\n── Test 7: Empty Index Retrieval ────────────────────")

    retriever = make_retriever_shared()
    # Don't index anything
    results = retriever.retrieve("some query", top_k=5)

    print(f"   Results on empty index: {len(results)}")
    assert results == []
    print("   ✅ Empty index returns [] not exception")


def test_index_empty_document():
    """Test indexing a document with no text returns 'empty' status."""
    print("\n── Test 8: Index Empty Document ─────────────────────")

    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata,
        DocumentType, ProcessingStatus
    )

    retriever = make_retriever_shared()

    empty_doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="empty.pdf",
            file_size_bytes=0,
            document_type=DocumentType.PDF,
            page_count=0,
        ),
        pages=[],
        full_text="",
        status=ProcessingStatus.EMPTY,
        error_message="No text extracted",
    )

    result = retriever.index_document(empty_doc, "empty_doc_id")

    print(f"   Status: {result['status']}")
    print(f"   Chunks: {result['chunks_indexed']}")
    print(f"   Error:  {result['error']}")

    assert result["status"] == "empty"
    assert result["chunks_indexed"] == 0
    print("   ✅ Empty document handled gracefully")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 7: RETRIEVER TESTS")
    print("=" * 55)

    test_index_document()
    test_retrieve_basic()
    test_retrieve_threshold()
    test_retrieve_empty_query_raises()
    test_delete_via_retriever()
    test_get_stats()
    test_empty_index_retrieve()
    test_index_empty_document()

    print("\n" + "=" * 55)
    print("   Phase 7 complete!")
    print("   Next: confirm all ✅ → Phase 8 (Ollama LLM)")
    print("=" * 55 + "\n")