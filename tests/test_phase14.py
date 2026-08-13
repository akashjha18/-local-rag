"""
test_phase14.py — Optimization Tests
=====================================
Run with:
    python tests/test_phase14.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.query_processor import preprocess_query, extract_keywords
from backend.utils.file_utils import (
    compute_content_hash,
    sanitize_filename,
    is_allowed_extension,
    extract_original_filename,
)
from backend.utils.logger import setup_logger

setup_logger(debug=False)


def test_query_preprocessing():
    print("\n── Test 1: Query Preprocessing ──────────────────────")

    test_cases = [
        ("  what is machine learning  ", "what is machine learning?"),
        ("what  is   faiss", "what is faiss?"),
        ("explain RAG", "explain RAG?"),
        ("WHAT IS PYTHON", "WHAT IS PYTHON?"),
    ]

    for raw, expected_contains in test_cases:
        result = preprocess_query(raw)
        print(f"   '{raw.strip()}' → '{result}'")
        assert len(result) >= 3, "Result too short"
        assert result == result.strip(), "Has extra whitespace"

    print("   ✅ Query preprocessing working")


def test_query_validation():
    print("\n── Test 2: Query Validation ─────────────────────────")

    invalid_queries = ["", "  ", "ab"]

    for q in invalid_queries:
        try:
            preprocess_query(q)
            print(f"   ❌ Should have rejected: '{q}'")
        except ValueError as e:
            print(f"   ✅ Rejected '{q}': {e}")


def test_keyword_extraction():
    print("\n── Test 3: Keyword Extraction ───────────────────────")

    query = "What machine learning algorithm was used in the study?"
    keywords = extract_keywords(query)
    print(f"   Query: '{query}'")
    print(f"   Keywords: {keywords}")

    assert "machine" in keywords
    assert "algorithm" in keywords
    assert "what" not in keywords
    assert "the" not in keywords
    print("   ✅ Keywords extracted correctly")


def test_content_hashing():
    print("\n── Test 4: Content Hashing ──────────────────────────")

    content1 = b"Hello, this is document content"
    content2 = b"Hello, this is document content"
    content3 = b"Different content entirely"

    hash1 = compute_content_hash(content1)
    hash2 = compute_content_hash(content2)
    hash3 = compute_content_hash(content3)

    print(f"   Hash1: {hash1[:16]}...")
    print(f"   Hash2: {hash2[:16]}...")
    print(f"   Hash3: {hash3[:16]}...")

    assert hash1 == hash2, "Same content should have same hash"
    assert hash1 != hash3, "Different content should have different hash"
    assert len(hash1) == 64, "SHA-256 should be 64 hex chars"
    print("   ✅ Content hashing working")


def test_filename_sanitization():
    print("\n── Test 5: Filename Sanitization ────────────────────")

    test_cases = [
        ("normal_file.pdf", True),
        ("file with spaces.pdf", True),
        ("../../../etc/passwd", True),
        ("file<script>.pdf", True),
        ("résumé.pdf", True),
    ]

    for filename, should_work in test_cases:
        result = sanitize_filename(filename)
        print(f"   '{filename}' → '{result}'")
        assert len(result) > 0, "Result should not be empty"
        assert ".." not in result, "Path traversal chars found"

    print("   ✅ Filename sanitization working")


def test_extension_validation():
    print("\n── Test 6: Extension Validation ─────────────────────")

    allowed = {".pdf", ".docx"}

    assert is_allowed_extension("report.pdf", allowed) is True
    assert is_allowed_extension("notes.docx", allowed) is True
    assert is_allowed_extension("image.png", allowed) is False
    assert is_allowed_extension("script.exe", allowed) is False
    assert is_allowed_extension("REPORT.PDF", allowed) is True  # case insensitive

    print("   ✅ Extension validation working")


def test_filename_extraction():
    print("\n── Test 7: Original Filename Extraction ─────────────")

    doc_id = "550e8400-e29b-41d4-a716-446655440000"
    stored = f"{doc_id}_AkashJha_Resume.pdf"

    original = extract_original_filename(stored, doc_id)
    print(f"   Stored:   '{stored}'")
    print(f"   Extracted: '{original}'")

    assert original == "AkashJha_Resume.pdf"
    print("   ✅ Filename extraction working")


def test_long_query_truncation():
    print("\n── Test 8: Long Query Handling ──────────────────────")

    long_query = "explain " + "machine learning " * 200
    print(f"   Input length: {len(long_query)} chars")

    result = preprocess_query(long_query)
    print(f"   Output length: {len(result)} chars")

    assert len(result) <= 2001, "Should be truncated"
    print("   ✅ Long query handled correctly")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 14: OPTIMIZATION TESTS")
    print("=" * 55)

    test_query_preprocessing()
    test_query_validation()
    test_keyword_extraction()
    test_content_hashing()
    test_filename_sanitization()
    test_extension_validation()
    test_filename_extraction()
    test_long_query_truncation()

    print("\n" + "=" * 55)
    print("   Phase 14 complete!")
    print("=" * 55 + "\n")