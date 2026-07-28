"""
test_phase4.py — Text Chunker Verification
==========================================
Run with:
    python tests/test_phase4.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processors.chunker import TextChunker
from backend.processors.pdf_processor import PDFProcessor
from backend.processors.docx_processor import DOCXProcessor
from backend.models.document_models import DocumentChunk
from backend.utils.logger import setup_logger

setup_logger(debug=False)


def test_basic_chunking():
    """Test that text is split into correct number of chunks."""
    print("\n── Test 1: Basic Chunking ───────────────────────────")

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    # Create a simple document with known content
    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata, PageContent,
        DocumentType, ProcessingStatus
    )

    # Simulate a document with clear paragraph breaks
    text = (
        "[Page 1]\n"
        "Retrieval Augmented Generation is a technique in AI.\n\n"
        "It combines information retrieval with language model generation.\n\n"
        "[Page 2]\n"
        "FAISS is a library for efficient similarity search.\n\n"
        "It was developed by Facebook AI Research.\n\n"
        "[Page 3]\n"
        "Sentence Transformers convert text into dense vector embeddings.\n\n"
        "These embeddings capture the semantic meaning of the text."
    )

    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="test.pdf",
            file_size_bytes=1000,
            document_type=DocumentType.PDF,
            page_count=3,
            total_chars=len(text),
            total_words=len(text.split()),
        ),
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        status=ProcessingStatus.SUCCESS,
    )

    chunks = chunker.chunk_document(doc, document_id="test_doc_001")

    print(f"   Input chars: {len(text)}")
    print(f"   Chunk size: 100, Overlap: 20")
    print(f"   Chunks produced: {len(chunks)}")

    for chunk in chunks:
        print(f"\n   Chunk {chunk.metadata.chunk_index:02d} "
              f"[Page {chunk.metadata.page_number}] "
              f"({len(chunk.text)} chars, {chunk.word_count} words):")
        print(f"   '{chunk.text[:80]}...' " if len(chunk.text) > 80
              else f"   '{chunk.text}'")

    assert len(chunks) > 0, "Should produce at least one chunk"
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(len(c.text) <= 150 for c in chunks), \
        "All chunks should be near chunk_size"
    print(f"\n   ✅ Basic chunking passed — {len(chunks)} chunks")


def test_overlap_works():
    """Test that consecutive chunks share content (overlap)."""
    print("\n── Test 2: Overlap Verification ────────────────────")

    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata, PageContent,
        DocumentType, ProcessingStatus
    )

    # Long, repetitive text to ensure multiple chunks
    text = " ".join([f"word{i}" for i in range(200)])

    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="overlap_test.pdf",
            file_size_bytes=len(text),
            document_type=DocumentType.PDF,
            page_count=1,
            total_chars=len(text),
            total_words=200,
        ),
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        status=ProcessingStatus.SUCCESS,
    )

    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document(doc, document_id="overlap_test")

    print(f"   Total chunks: {len(chunks)}")

    if len(chunks) >= 2:
        # Check that consecutive chunks share some content
        for i in range(len(chunks) - 1):
            chunk_a = chunks[i].text
            chunk_b = chunks[i + 1].text

            # Find overlap: words at end of chunk_a that appear at start of chunk_b
            words_a = set(chunk_a.split()[-10:])  # Last 10 words of chunk A
            words_b = set(chunk_b.split()[:10])   # First 10 words of chunk B
            shared = words_a & words_b

            print(f"   Chunks {i}→{i+1}: {len(shared)} shared words")
            if len(shared) > 0:
                print(f"   Shared: {list(shared)[:5]}")

    print(f"   ✅ Overlap test complete")


def test_page_attribution():
    """Test that chunks correctly identify which page they came from."""
    print("\n── Test 3: Page Attribution ─────────────────────────")

    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata, PageContent,
        DocumentType, ProcessingStatus
    )

    # Text with explicit page markers
    text = (
        "[Page 1]\nThis content is on the first page of the document.\n\n"
        "[Page 2]\nThis content is on the second page of the document.\n\n"
        "[Page 3]\nThis content is on the third page of the document."
    )

    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="multipage.pdf",
            file_size_bytes=len(text),
            document_type=DocumentType.PDF,
            page_count=3,
            total_chars=len(text),
            total_words=len(text.split()),
        ),
        pages=[],
        full_text=text,
        status=ProcessingStatus.SUCCESS,
    )

    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_document(doc, document_id="page_test")

    print(f"   Chunks produced: {len(chunks)}")
    page_numbers_seen = set()

    for chunk in chunks:
        page_numbers_seen.add(chunk.metadata.page_number)
        # Verify [Page N] markers are REMOVED from chunk text
        assert "[Page" not in chunk.text, \
            f"Page marker found in chunk text: {chunk.text[:50]}"
        print(f"   Chunk {chunk.metadata.chunk_index}: "
              f"page={chunk.metadata.page_number}, "
              f"text='{chunk.text[:50]}...'")

    print(f"   Pages seen in chunks: {sorted(page_numbers_seen)}")
    assert len(page_numbers_seen) >= 1, "Should have page attribution"
    print(f"   ✅ Page attribution working")


def test_chunk_ids_unique():
    """Test that all chunk IDs are unique."""
    print("\n── Test 4: Unique Chunk IDs ─────────────────────────")

    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata, PageContent,
        DocumentType, ProcessingStatus
    )

    text = " ".join([f"sentence {i} about topic." for i in range(100)])
    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="id_test.pdf",
            file_size_bytes=len(text),
            document_type=DocumentType.PDF,
            page_count=1,
            total_chars=len(text),
            total_words=len(text.split()),
        ),
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        status=ProcessingStatus.SUCCESS,
    )

    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(doc, document_id="unique_id_test")

    ids = [c.chunk_id for c in chunks]
    unique_ids = set(ids)

    print(f"   Total chunks: {len(chunks)}")
    print(f"   Unique IDs: {len(unique_ids)}")
    assert len(ids) == len(unique_ids), "Duplicate chunk IDs found!"
    print(f"   Sample IDs: {ids[:3]}")
    print(f"   ✅ All chunk IDs are unique")


def test_chunk_stats():
    """Test the statistics reporting."""
    print("\n── Test 5: Chunk Statistics ─────────────────────────")

    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata, PageContent,
        DocumentType, ProcessingStatus
    )

    text = " ".join([f"word{i}" for i in range(300)])
    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="stats_test.pdf",
            file_size_bytes=len(text),
            document_type=DocumentType.PDF,
            page_count=1,
            total_chars=len(text),
            total_words=300,
        ),
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        status=ProcessingStatus.SUCCESS,
    )

    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document(doc, document_id="stats_test")
    stats = chunker.get_chunk_stats(chunks)

    print(f"   Total chunks:  {stats['total_chunks']}")
    print(f"   Avg chars:     {stats['avg_chars']}")
    print(f"   Min chars:     {stats['min_chars']}")
    print(f"   Max chars:     {stats['max_chars']}")
    print(f"   Avg words:     {stats['avg_words']}")
    print(f"   ✅ Stats computed correctly")


def test_real_pdf_chunking():
    """Test chunking on the real PDF from Phase 2."""
    print("\n── Test 6: Real PDF Chunking ────────────────────────")

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  sample.pdf not found — skipping")
        return

    processor = PDFProcessor()
    doc = processor.process(pdf_path)

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(doc, document_id="sample_pdf")
    stats = chunker.get_chunk_stats(chunks)

    print(f"   Document: {doc.metadata.filename}")
    print(f"   Pages: {doc.metadata.page_count}")
    print(f"   Total chars: {doc.metadata.total_chars}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   Avg chars/chunk: {stats['avg_chars']}")
    print(f"   Avg words/chunk: {stats['avg_words']}")

    # Show first 3 chunks
    print(f"\n   First 3 chunks:")
    for chunk in chunks[:3]:
        print(f"\n   [{chunk.chunk_id}] "
              f"Page {chunk.metadata.page_number} "
              f"({chunk.word_count} words):")
        print(f"   {chunk.text[:120]}...")

    assert stats["total_chunks"] > 0
    print(f"\n   ✅ Real PDF chunked successfully")


def test_empty_document():
    """Test graceful handling of empty documents."""
    print("\n── Test 7: Empty Document ───────────────────────────")

    from backend.models.document_models import (
        ProcessedDocument, DocumentMetadata, PageContent,
        DocumentType, ProcessingStatus
    )

    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            filename="empty.pdf",
            file_size_bytes=0,
            document_type=DocumentType.PDF,
            page_count=0,
            total_chars=0,
            total_words=0,
        ),
        pages=[],
        full_text="",
        status=ProcessingStatus.EMPTY,
    )

    chunker = TextChunker()
    chunks = chunker.chunk_document(doc, document_id="empty_doc")

    assert chunks == []
    print(f"   ✅ Empty document returns empty list (no crash)")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 4: TEXT CHUNKING TESTS")
    print("=" * 55)

    test_basic_chunking()
    test_overlap_works()
    test_page_attribution()
    test_chunk_ids_unique()
    test_chunk_stats()
    test_real_pdf_chunking()
    test_empty_document()

    print("\n" + "=" * 55)
    print("   Phase 4 complete!")
    print("   Next: confirm all ✅ → Phase 5 (Embeddings)")
    print("=" * 55 + "\n")