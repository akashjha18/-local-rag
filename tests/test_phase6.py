"""
test_phase6.py — FAISS Vector Store Verification
=================================================
Run with:
    python tests/test_phase6.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.vector_store import FAISSVectorStore
from backend.rag.embeddings import EmbeddingService
from backend.models.document_models import SearchResult
from backend.utils.logger import setup_logger

setup_logger(debug=False)

# ── Test paths (separate from production data) ─────────────────────
TEST_INDEX_PATH = "data/vector_store/test_index.faiss"
TEST_META_PATH  = "data/vector_store/test_metadata.json"


def make_test_store() -> FAISSVectorStore:
    """Create a fresh test store (cleans up previous test data)."""
    store = FAISSVectorStore(
        index_path=TEST_INDEX_PATH,
        metadata_path=TEST_META_PATH,
        dimension=384,
    )
    store.reset()  # Start clean
    return store


def make_test_chunks_and_vectors(n: int, embedding_service: EmbeddingService):
    """Create n test DocumentChunks with real embeddings."""
    from backend.models.document_models import (
        DocumentChunk, ChunkMetadata
    )

    texts = [
        "Python is a high-level programming language used in AI.",
        "FAISS enables fast similarity search over large vector sets.",
        "Sentence Transformers encode text into dense embeddings.",
        "FastAPI is a modern Python web framework for building APIs.",
        "RAG combines retrieval with language model generation.",
        "SQLite is a lightweight relational database for local storage.",
        "React is a JavaScript library for building user interfaces.",
        "Ollama runs large language models locally on your machine.",
        "Jaipur is the capital city of Rajasthan in India.",
        "JECRC University is located in Jaipur, Rajasthan.",
    ][:n]

    chunks = [
        DocumentChunk(
            chunk_id=f"test_doc_chunk_{i:04d}",
            text=text,
            metadata=ChunkMetadata(
                document_id="test_document",
                filename="test.pdf",
                document_type="pdf",
                page_number=(i // 3) + 1,
                chunk_index=i,
                total_chunks=n,
                char_start=i * 100,
                char_end=(i + 1) * 100,
            ),
        )
        for i, text in enumerate(texts)
    ]

    vectors = embedding_service.embed_chunks(chunks, show_progress=False)
    return chunks, vectors


print("\nLoading embedding model...")
embedding_service = EmbeddingService()


def test_add_and_count():
    """Test adding chunks and counting vectors."""
    print("\n── Test 1: Add Chunks ───────────────────────────────")

    store = make_test_store()
    assert store.total_vectors == 0
    assert store.is_empty

    chunks, vectors = make_test_chunks_and_vectors(5, embedding_service)
    added = store.add_chunks(chunks, vectors)

    print(f"   Chunks added: {added}")
    print(f"   Total vectors: {store.total_vectors}")
    print(f"   Index size: {store.index_size_mb:.4f} MB")
    print(f"   Store repr: {store}")

    assert added == 5
    assert store.total_vectors == 5
    assert not store.is_empty
    print("   ✅ Add and count working")


def test_search_returns_results():
    """Test that search returns relevant results."""
    print("\n── Test 2: Similarity Search ────────────────────────")

    store = make_test_store()
    chunks, vectors = make_test_chunks_and_vectors(10, embedding_service)
    store.add_chunks(chunks, vectors)

    # Query about Python
    query_vec = embedding_service.embed_query("Python programming language")
    results = store.search(query_vec, top_k=3)

    print(f"   Query: 'Python programming language'")
    print(f"   Results returned: {len(results)}")
    print()

    for r in results:
        print(f"   Score: {r.score:.4f} | {r.source_label} | {r.confidence_percent}%")
        print(f"   Text:  '{r.text[:70]}...'")
        print()

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].score >= results[1].score  # Must be sorted by score
    assert results[0].score > 0.4  # Top result should be highly relevant
    print("   ✅ Search returning ranked results")


def test_score_threshold():
    """Test that score_threshold filters low-quality results."""
    print("\n── Test 3: Score Threshold Filter ───────────────────")

    store = make_test_store()
    chunks, vectors = make_test_chunks_and_vectors(10, embedding_service)
    store.add_chunks(chunks, vectors)

    query_vec = embedding_service.embed_query("quantum physics nuclear reactor")

    results_no_filter = store.search(query_vec, top_k=5, score_threshold=0.0)
    results_filtered  = store.search(query_vec, top_k=5, score_threshold=0.4)

    print(f"   Query: 'quantum physics nuclear reactor' (irrelevant)")
    print(f"   Results without threshold: {len(results_no_filter)}")
    print(f"   Results with threshold=0.4: {len(results_filtered)}")

    if results_no_filter:
        print(f"   Highest score (no filter): {results_no_filter[0].score:.4f}")
    if results_filtered:
        print(f"   All filtered results have score > 0.4: "
              f"{all(r.score > 0.4 for r in results_filtered)}")

    assert len(results_filtered) <= len(results_no_filter)
    print("   ✅ Score threshold filtering working")


def test_persistence():
    """Test that index survives save and reload."""
    print("\n── Test 4: Persistence (Save & Load) ────────────────")

    # Create and populate store
    store1 = make_test_store()
    chunks, vectors = make_test_chunks_and_vectors(5, embedding_service)
    store1.add_chunks(chunks, vectors)
    initial_count = store1.total_vectors

    print(f"   Saved {initial_count} vectors to disk")

    # Load it fresh from disk
    store2 = FAISSVectorStore(
        index_path=TEST_INDEX_PATH,
        metadata_path=TEST_META_PATH,
        dimension=384,
    )

    print(f"   Loaded {store2.total_vectors} vectors from disk")
    assert store2.total_vectors == initial_count, \
        f"Expected {initial_count}, got {store2.total_vectors}"

    # Search still works after reload
    query_vec = embedding_service.embed_query("Python programming")
    results = store2.search(query_vec, top_k=2)
    assert len(results) > 0, "Search should work after reload"
    print(f"   Search after reload: {len(results)} results")
    print(f"   Top result: '{results[0].text[:50]}...' (score={results[0].score:.4f})")
    print("   ✅ Persistence working")


def test_delete_document():
    """Test removing a document's chunks from the index."""
    print("\n── Test 5: Document Deletion ────────────────────────")

    from backend.models.document_models import DocumentChunk, ChunkMetadata

    store = make_test_store()

    # Add two different documents
    doc_a_texts = ["Document A content about machine learning.", "More ML content here."]
    doc_b_texts = ["Document B content about cooking recipes.", "More cooking tips."]

    def make_doc_chunks(texts, doc_id, filename):
        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i}",
                text=text,
                metadata=ChunkMetadata(
                    document_id=doc_id,
                    filename=filename,
                    document_type="pdf",
                    page_number=1,
                    chunk_index=i,
                    total_chunks=len(texts),
                    char_start=0,
                    char_end=len(text),
                ),
            )
            for i, text in enumerate(texts)
        ]
        vectors = embedding_service.embed_chunks(chunks, show_progress=False)
        return chunks, vectors

    chunks_a, vectors_a = make_doc_chunks(doc_a_texts, "doc_a", "ml_paper.pdf")
    chunks_b, vectors_b = make_doc_chunks(doc_b_texts, "doc_b", "recipes.pdf")

    store.add_chunks(chunks_a, vectors_a)
    store.add_chunks(chunks_b, vectors_b)

    print(f"   Total after adding both docs: {store.total_vectors}")
    assert store.total_vectors == 4

    # Delete doc_a
    removed = store.delete_document("doc_a")
    print(f"   Chunks removed: {removed}")
    print(f"   Total after deletion: {store.total_vectors}")

    assert removed == 2
    assert store.total_vectors == 2

    # Verify doc_b still searchable
    query_vec = embedding_service.embed_query("cooking recipes food")
    results = store.search(query_vec, top_k=5)
    filenames = [r.filename for r in results]

    print(f"   Remaining results: {len(results)}")
    print(f"   Filenames found: {set(filenames)}")

    assert all(r.filename == "recipes.pdf" for r in results), \
        "Should only find doc_b after deleting doc_a"
    print("   ✅ Document deletion working")


def test_source_label_and_confidence():
    """Test the convenience properties on SearchResult."""
    print("\n── Test 6: SearchResult Properties ──────────────────")

    store = make_test_store()
    chunks, vectors = make_test_chunks_and_vectors(5, embedding_service)
    store.add_chunks(chunks, vectors)

    query_vec = embedding_service.embed_query("FAISS vector search")
    results = store.search(query_vec, top_k=1)

    assert len(results) > 0
    r = results[0]

    print(f"   source_label:       '{r.source_label}'")
    print(f"   confidence_percent: {r.confidence_percent}%")
    print(f"   score:              {r.score:.4f}")

    assert "Page" in r.source_label
    assert 0 <= r.confidence_percent <= 100
    print("   ✅ SearchResult properties correct")


def test_full_pipeline_e2e():
    """
    Full end-to-end: PDF → chunks → embeddings → FAISS → search.
    The complete pipeline up to Phase 6.
    """
    print("\n── Test 7: Full Pipeline E2E ─────────────────────────")

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  sample.pdf not found — skipping")
        return

    from backend.processors.pdf_processor import PDFProcessor
    from backend.processors.chunker import TextChunker

    store = make_test_store()

    # Step 1: Extract
    doc = PDFProcessor().process(pdf_path)
    print(f"   Extracted: {doc.metadata.total_chars} chars")

    # Step 2: Chunk
    chunks = TextChunker(chunk_size=512, chunk_overlap=50).chunk_document(
        doc, document_id="e2e_test_pdf"
    )
    print(f"   Chunked: {len(chunks)} chunks")

    # Step 3: Embed
    vectors = embedding_service.embed_chunks(chunks, show_progress=False)
    print(f"   Embedded: {vectors.shape}")

    # Step 4: Store
    store.add_chunks(chunks, vectors)
    print(f"   Stored: {store.total_vectors} vectors")

    # Step 5: Search
    query = "What machine learning algorithm was used?"
    query_vec = embedding_service.embed_query(query)
    results = store.search(query_vec, top_k=3)

    print(f"\n   Query: '{query}'")
    print(f"   Results:")
    for r in results:
        print(f"   [{r.confidence_percent}%] {r.source_label}")
        print(f"   '{r.text[:80]}...'")
        print()

    assert len(results) > 0
    assert results[0].score > 0.2

    # Step 6: Verify persistence
    docs = store.get_indexed_documents()
    print(f"   Indexed documents: {docs}")
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == len(chunks)

    print("   ✅ Full pipeline E2E complete")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 6: FAISS VECTOR STORE TESTS")
    print("=" * 55)

    test_add_and_count()
    test_search_returns_results()
    test_score_threshold()
    test_persistence()
    test_delete_document()
    test_source_label_and_confidence()
    test_full_pipeline_e2e()

    print("\n" + "=" * 55)
    print("   Phase 6 complete!")
    print("   Next: confirm all ✅ → Phase 7 (Retriever)")
    print("=" * 55 + "\n")