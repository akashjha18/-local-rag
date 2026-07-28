"""
test_phase8.py — Ollama LLM Integration Tests
==============================================
Requires: ollama running + mistral pulled
  ollama list  → should show mistral:latest

Run with:
    python tests/test_phase8.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.llm import OllamaLLM
from backend.rag.embeddings import EmbeddingService
from backend.rag.vector_store import FAISSVectorStore
from backend.rag.retriever import Retriever
from backend.processors.chunker import TextChunker
from backend.utils.logger import setup_logger

setup_logger(debug=False)


def test_ollama_available():
    """Verify Ollama is running and mistral is loaded."""
    print("\n── Test 1: Ollama Available ──────────────────────────")

    llm = OllamaLLM()
    available = llm.is_available()
    models = llm.list_models()

    print(f"   Ollama available: {available}")
    print(f"   Models installed: {models}")
    print(f"   Target model:     {llm.model}")

    assert available, (
        "Ollama not available! Run: ollama serve\n"
        "And ensure mistral is pulled: ollama pull mistral"
    )
    print("   ✅ Ollama is running with mistral")


def test_answer_from_context():
    """Test basic answer generation from provided context."""
    print("\n── Test 2: Answer From Context ──────────────────────")

    from backend.models.document_models import SearchResult

    llm = OllamaLLM()

    # Synthetic context — no real documents needed
    fake_chunks = [
        SearchResult(
            chunk_id="chunk_001",
            text=(
                "FAISS (Facebook AI Similarity Search) is a library "
                "developed by Meta AI Research for efficient similarity "
                "search and clustering of dense vectors. It supports "
                "both CPU and GPU implementations."
            ),
            score=0.91,
            filename="faiss_docs.pdf",
            document_id="doc_001",
            page_number=1,
            chunk_index=0,
            document_type="pdf",
        ),
        SearchResult(
            chunk_id="chunk_002",
            text=(
                "FAISS IndexFlatIP performs exact inner product search. "
                "For normalized vectors, inner product equals cosine "
                "similarity, making it ideal for semantic search."
            ),
            score=0.85,
            filename="faiss_docs.pdf",
            document_id="doc_001",
            page_number=2,
            chunk_index=1,
            document_type="pdf",
        ),
    ]

    query = "What is FAISS and who developed it?"
    print(f"   Query: '{query}'")
    print(f"   Context chunks: {len(fake_chunks)}")
    print(f"   Calling Ollama (may take 10-30 seconds)...")

    response = llm.generate_answer(query, fake_chunks)

    print(f"\n   ── LLM Response ──────────────────────────────────")
    print(f"   Model:      {response.model}")
    print(f"   Success:    {response.success}")
    print(f"   Time:       {response.generation_time:.2f}s")
    print(f"   Tokens:     {response.total_tokens}")
    print(f"\n   Answer:\n   {response.answer}")

    assert response.success, f"LLM failed: {response.error}"
    assert len(response.answer) > 20, "Answer too short"
    assert response.sources == fake_chunks
    print(f"\n   ✅ LLM generated answer successfully")


def test_no_context_response():
    """Test behavior when no context chunks are provided."""
    print("\n── Test 3: No Context Handling ──────────────────────")

    llm = OllamaLLM()
    response = llm.generate_answer("What is the meaning of life?", [])

    print(f"   Answer: {response.answer}")
    assert response.success
    assert "cannot" in response.answer.lower() or "no" in response.answer.lower()
    print("   ✅ No context handled gracefully")


def test_hallucination_prevention():
    """
    Critical test: LLM should NOT answer from its own knowledge
    when the context doesn't contain the answer.
    """
    print("\n── Test 4: Hallucination Prevention ─────────────────")

    from backend.models.document_models import SearchResult

    llm = OllamaLLM()

    # Context about cooking — completely unrelated to the question
    unrelated_chunks = [
        SearchResult(
            chunk_id="c1",
            text="To make pasta, boil water with salt. Add pasta and cook for 10 minutes.",
            score=0.42,
            filename="recipes.pdf",
            document_id="d1",
            page_number=1,
            chunk_index=0,
            document_type="pdf",
        ),
    ]

    # Question about AI — not in context
    query = "What is the learning rate in neural network training?"
    print(f"   Query: '{query}'")
    print(f"   Context: about cooking (irrelevant)")
    print(f"   Calling Ollama...")

    response = llm.generate_answer(query, unrelated_chunks)

    print(f"\n   Answer: {response.answer}")

    # The LLM should say it can't find the info, not hallucinate
    answer_lower = response.answer.lower()
    cannot_answer = any(phrase in answer_lower for phrase in [
        "cannot", "not find", "not in", "don't", "no information",
        "context does not", "not mentioned", "not provided",
        "based on the context", "the context"
    ])

    print(f"   Refused to hallucinate: {cannot_answer}")
    if not cannot_answer:
        print(f"   ⚠️  LLM may have hallucinated — review the answer above")
    else:
        print(f"   ✅ LLM correctly refused to answer from missing context")


def test_full_rag_pipeline():
    """
    THE BIG TEST: Upload PDF → Index → Query → LLM Answer
    Complete end-to-end RAG pipeline.
    """
    print("\n── Test 5: Full RAG Pipeline ─────────────────────────")

    pdf_path = Path("data/documents/sample.pdf")
    if not pdf_path.exists():
        print("   ⏭️  sample.pdf not found")
        return

    print("   Building pipeline components...")

    # ── Initialize all components ──────────────────────────────────
    emb_service = EmbeddingService()
    vector_store = FAISSVectorStore(
        index_path="data/vector_store/rag_test.faiss",
        metadata_path="data/vector_store/rag_test_metadata.json",
        dimension=384,
    )
    vector_store.reset()

    retriever = Retriever(
        emb_service,
        vector_store,
        TextChunker(chunk_size=512, chunk_overlap=50),
    )
    llm = OllamaLLM()

    # ── Step 1: Index the document ─────────────────────────────────
    from backend.processors.pdf_processor import PDFProcessor
    doc = PDFProcessor().process(pdf_path)
    index_result = retriever.index_document(doc, "sample_pdf_rag")

    print(f"\n   Indexed: {index_result['chunks_indexed']} chunks")

    # ── Step 2: Ask questions ──────────────────────────────────────
    questions = [
        "What is the main topic of this document?",
        "What dataset was used in the study?",
        "What methodology or algorithm was applied?",
    ]

    for question in questions:
        print(f"\n   {'─'*50}")
        print(f"   Q: {question}")

        # Retrieve relevant chunks
        chunks = retriever.retrieve(question, top_k=3, score_threshold=0.2)
        print(f"   Retrieved: {len(chunks)} chunks | "
              f"Top score: {chunks[0].score:.3f}" if chunks else
              f"   Retrieved: 0 chunks")

        if not chunks:
            print("   A: No relevant context found")
            continue

        # Generate answer
        print(f"   Generating answer (please wait)...")
        response = llm.generate_answer(question, chunks)

        print(f"\n   A: {response.answer}")
        print(f"\n   Sources:")
        for src in response.unique_sources:
            print(f"     • {src.source_label} [{src.confidence_percent}%]")
        print(f"   Time: {response.generation_time:.1f}s | "
              f"Tokens: {response.total_tokens}")

    print(f"\n   ✅ Full RAG pipeline working end-to-end")


def test_source_citation():
    """Verify sources are correctly attached to responses."""
    print("\n── Test 6: Source Citations ──────────────────────────")

    from backend.models.document_models import SearchResult

    llm = OllamaLLM()

    chunks = [
        SearchResult(
            chunk_id=f"c{i}",
            text=f"This is content from page {i} about machine learning.",
            score=0.9 - (i * 0.1),
            filename="ml_paper.pdf",
            document_id="doc1",
            page_number=i,
            chunk_index=i,
            document_type="pdf",
        )
        for i in range(1, 4)
    ]

    response = llm.generate_answer(
        "What is this document about?", chunks
    )

    print(f"   Total sources:  {len(response.sources)}")
    print(f"   Unique sources: {len(response.unique_sources)}")

    for src in response.unique_sources:
        print(f"   • {src.source_label} [{src.confidence_percent}%]")

    assert len(response.sources) == 3
    print("   ✅ Sources correctly attached to response")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 8: OLLAMA LLM TESTS")
    print("=" * 55)
    print("   ⚠️  Tests will call Mistral locally.")
    print("   Each LLM call takes 10-60 seconds on CPU.")
    print("=" * 55)

    test_ollama_available()
    test_answer_from_context()
    test_no_context_response()
    test_hallucination_prevention()
    test_full_rag_pipeline()
    test_source_citation()

    print("\n" + "=" * 55)
    print("   Phase 8 complete!")
    print("   Next: confirm all ✅ → Phase 9 (RAG Pipeline)")
    print("=" * 55 + "\n")