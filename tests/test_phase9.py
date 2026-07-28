"""
test_phase9.py — RAG Pipeline Verification
===========================================
Tests the complete pipeline through the single RAGPipeline class.

Run with:
    python tests/test_phase9.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.pipeline import RAGPipeline
from backend.models.document_models import RAGResponse, IngestResult
from backend.utils.logger import setup_logger

setup_logger(debug=False)


print("\nInitializing RAG Pipeline (loads all components)...")
print("This takes ~5 seconds on first run...")

pipeline = RAGPipeline()


def test_pipeline_initializes():
    """Verify all components loaded correctly."""
    print("\n── Test 1: Pipeline Initialization ──────────────────")

    stats = pipeline.get_stats()

    print(f"   Pipeline ready:    {stats['pipeline_ready']}")
    print(f"   LLM available:     {stats['llm_available']}")
    print(f"   LLM model:         {stats['llm_model']}")
    print(f"   Embedding model:   {stats['embedding_model']}")
    print(f"   Total vectors:     {stats['total_vectors']}")
    print(f"   Chunk size:        {stats['chunk_size']}")
    print(f"   Top K:             {stats['top_k']}")

    assert stats["pipeline_ready"] is True
    assert stats["llm_available"] is True
    assert stats["embedding_model"] == "all-MiniLM-L6-v2"
    print("   ✅ Pipeline initialized correctly")


def test_ingest_pdf():
    """Test PDF ingestion through the pipeline."""
    print("\n── Test 2: Ingest PDF ───────────────────────────────")

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  sample.pdf not found")
        return

    # Clean slate
    pipeline.vector_store.reset()

    result = pipeline.ingest(pdf_path, document_id="test_pipeline_pdf")

    print(f"   Filename:       {result.filename}")
    print(f"   Status:         {result.status}")
    print(f"   Chunks indexed: {result.chunks_indexed}")
    print(f"   Pages:          {result.page_count}")
    print(f"   Words:          {result.word_count}")
    print(f"   Time:           {result.processing_time:.2f}s")
    print(f"   Document ID:    {result.document_id}")
    print(f"   Error:          {result.error}")

    assert isinstance(result, IngestResult)
    assert result.status == "success"
    assert result.chunks_indexed > 0
    assert result.error is None
    print("   ✅ PDF ingested successfully")


def test_ingest_docx():
    """Test DOCX ingestion through the pipeline."""
    print("\n── Test 3: Ingest DOCX ──────────────────────────────")

    docx_path = Path("data/documents/test_basic.docx")
    if not docx_path.exists():
        print("   ⏭️  test_basic.docx not found")
        return

    result = pipeline.ingest(docx_path, document_id="test_pipeline_docx")

    print(f"   Filename: {result.filename}")
    print(f"   Status:   {result.status}")
    print(f"   Chunks:   {result.chunks_indexed}")
    print(f"   Time:     {result.processing_time:.2f}s")

    assert result.status == "success"
    assert result.chunks_indexed > 0
    print("   ✅ DOCX ingested successfully")


def test_ask_question():
    """Test the full ask() pipeline."""
    print("\n── Test 4: Ask Question ─────────────────────────────")

    # Ensure something is indexed
    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  No document to query")
        return

    if pipeline.vector_store.is_empty:
        pipeline.ingest(pdf_path, document_id="ask_test_pdf")

    query = "What recommendation algorithm was used in this research?"
    print(f"   Query: '{query}'")
    print(f"   Calling pipeline (please wait)...")

    response = pipeline.ask(query, top_k=3)

    print(f"\n   ── Response ────────────────────────────────────")
    print(f"   Success:          {response.success}")
    print(f"   Chunks retrieved: {response.chunks_retrieved}")
    print(f"   Retrieval time:   {response.retrieval_time:.3f}s")
    print(f"   Generation time:  {response.generation_time:.1f}s")
    print(f"   Total time:       {response.total_time:.1f}s")
    print(f"\n   Answer:\n   {response.answer}")
    print(f"\n   Sources:")
    for src in response.sources:
        print(f"     • {src.source_label} [{src.confidence_percent}%]")

    assert isinstance(response, RAGResponse)
    assert response.success
    assert len(response.answer) > 10
    print("\n   ✅ ask() pipeline working end-to-end")


def test_ask_empty_index():
    """Test ask() on empty index returns graceful response."""
    print("\n── Test 5: Ask on Empty Index ───────────────────────")

    # Reset to empty
    pipeline.vector_store.reset()

    response = pipeline.ask("What is machine learning?")

    print(f"   Answer: {response.answer[:100]}")
    assert response.success is True
    assert "cannot" in response.answer.lower() or \
           "no relevant" in response.answer.lower() or \
           "upload" in response.answer.lower()
    print("   ✅ Empty index handled gracefully")


def test_unsupported_file_type():
    """Test that unsupported file types return error."""
    print("\n── Test 6: Unsupported File Type ────────────────────")

    fake_txt = Path("data/documents/fake.txt")
    fake_txt.write_text("hello world")

    result = pipeline.ingest(fake_txt)

    print(f"   Status: {result.status}")
    print(f"   Error:  {result.error}")

    assert result.status == "failed"
    assert "Unsupported" in result.error
    print("   ✅ Unsupported file type rejected correctly")

    fake_txt.unlink()


def test_delete_document():
    """Test document deletion through pipeline."""
    print("\n── Test 7: Delete Document ──────────────────────────")

    pipeline.vector_store.reset()

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  No document to delete")
        return

    # Ingest
    result = pipeline.ingest(pdf_path, document_id="delete_test_doc")
    print(f"   Indexed: {result.chunks_indexed} chunks")

    # Verify indexed
    docs = pipeline.get_indexed_documents()
    assert len(docs) == 1

    # Delete
    removed = pipeline.delete_document("delete_test_doc")
    print(f"   Removed: {removed} chunks")

    # Verify gone
    docs_after = pipeline.get_indexed_documents()
    print(f"   Documents remaining: {len(docs_after)}")

    assert removed == result.chunks_indexed
    assert len(docs_after) == 0
    print("   ✅ Document deleted through pipeline")


def test_pipeline_is_ready_property():
    """Test the is_ready property."""
    print("\n── Test 8: is_ready Property ────────────────────────")

    # Empty index
    pipeline.vector_store.reset()
    print(f"   is_ready (empty):    {pipeline.is_ready}")
    assert pipeline.is_ready is False

    # After ingestion
    pdf_path = Path("data/documents/sample.pdf")
    if pdf_path.exists():
        pipeline.ingest(pdf_path, document_id="ready_test")
        print(f"   is_ready (indexed):  {pipeline.is_ready}")
        assert pipeline.is_ready is True

    print("   ✅ is_ready property working correctly")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 9: RAG PIPELINE TESTS")
    print("=" * 55)

    test_pipeline_initializes()
    test_ingest_pdf()
    test_ingest_docx()
    test_ask_question()
    test_ask_empty_index()
    test_unsupported_file_type()
    test_delete_document()
    test_pipeline_is_ready_property()

    print("\n" + "=" * 55)
    print("   Phase 9 complete!")
    print("   Next: confirm all ✅ → Phase 10 (FastAPI Backend)")
    print("=" * 55 + "\n")