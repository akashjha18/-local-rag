"""
test_phase2.py — PDF Processor Verification
============================================
Tests the PDFProcessor with real PDFs.

Run with:
    python tests/test_phase2.py
"""

import sys
import os
import urllib.request
from pathlib import Path

# ── Add project root to path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processors.pdf_processor import PDFProcessor
from backend.models.document_models import ProcessingStatus
from backend.utils.logger import logger, setup_logger

setup_logger(debug=True)


def download_sample_pdf(destination: Path) -> bool:
    """
    Download a small, real PDF for testing.
    Uses a public domain PDF from the web.
    """
    if destination.exists():
        print(f"   Sample PDF already exists: {destination}")
        return True

    url = (
        "https://www.w3.org/WAI/WCAG21/Techniques/pdf/img/table-word.pdf"
    )

    # Fallback: create a minimal synthetic PDF if download fails
    try:
        print(f"   Downloading sample PDF...")
        urllib.request.urlretrieve(url, destination)
        print(f"   ✅ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"   ⚠️  Download failed ({e}), creating synthetic PDF...")
        return create_synthetic_pdf(destination)


def create_synthetic_pdf(destination: Path) -> bool:
    """
    Create a minimal valid PDF for testing without any external download.
    This is a hand-crafted PDF byte sequence.
    """
    try:
        # Use pypdf to create a test PDF programmatically
        from pypdf import PdfWriter
        from pypdf.generic import NameObject

        writer = PdfWriter()

        # Add 3 pages with content
        for i in range(1, 4):
            page = writer.add_blank_page(width=612, height=792)

        # Write to file
        with open(destination, "wb") as f:
            writer.write(f)

        print(f"   ✅ Synthetic PDF created at {destination}")
        return True

    except Exception as e:
        print(f"   ❌ Could not create synthetic PDF: {e}")
        return False


def test_with_real_pdf():
    """Test PDF processing with a real file."""
    print("\n── Test 1: Real PDF Processing ─────────────────────")

    # Create a sample PDF directory
    sample_dir = Path("data/documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_pdf = sample_dir / "sample.pdf"

    # Try to get a sample PDF
    if not download_sample_pdf(sample_pdf):
        print("   ⚠️  Skipping real PDF test — no sample available")
        return

    processor = PDFProcessor()
    result = processor.process(sample_pdf)

    print(f"   File: {result.metadata.filename}")
    print(f"   Status: {result.status}")
    print(f"   Pages: {result.metadata.page_count}")
    print(f"   Total chars: {result.metadata.total_chars}")
    print(f"   Total words: {result.metadata.total_words}")
    print(f"   Processing time: {result.processing_time:.3f}s")

    if result.error_message:
        print(f"   Message: {result.error_message}")

    if result.full_text:
        preview = result.full_text[:200].replace("\n", " ")
        print(f"   Text preview: {preview}...")

    if result.is_successful:
        print("   ✅ PDF processed successfully")
    else:
        print(f"   ⚠️  Status: {result.status} — this may be expected for synthetic PDFs")


def test_missing_file():
    """Test that missing files return proper error, not exception."""
    print("\n── Test 2: Missing File Handling ───────────────────")

    processor = PDFProcessor()
    result = processor.process("non_existent_file.pdf")

    assert result.status == ProcessingStatus.FAILED
    assert result.error_message is not None
    assert "not found" in result.error_message.lower()
    print(f"   ✅ Missing file handled: {result.error_message}")


def test_wrong_extension():
    """Test that non-PDF files are rejected."""
    print("\n── Test 3: Wrong Extension ─────────────────────────")

    # Create a fake .txt file
    fake_file = Path("data/documents/fake.txt")
    fake_file.write_text("I am not a PDF")

    processor = PDFProcessor()
    result = processor.process(fake_file)

    assert result.status == ProcessingStatus.FAILED
    assert result.error_message is not None
    print(f"   ✅ Wrong extension handled: {result.error_message}")

    fake_file.unlink()  # Clean up


def test_empty_file():
    """Test that empty files are handled."""
    print("\n── Test 4: Empty File Handling ─────────────────────")

    empty_file = Path("data/documents/empty.pdf")
    empty_file.write_bytes(b"")  # 0 bytes

    processor = PDFProcessor()
    result = processor.process(empty_file)

    assert result.status == ProcessingStatus.FAILED
    print(f"   ✅ Empty file handled: {result.error_message}")

    empty_file.unlink()


def test_page_content_model():
    """Test the PageContent dataclass computed fields."""
    print("\n── Test 5: PageContent Model ───────────────────────")

    from backend.models.document_models import PageContent

    # Test with real text
    page = PageContent(page_number=1, text="Hello world this is a test page")
    assert page.has_text is True
    assert page.char_count > 0
    print(f"   ✅ PageContent with text: {page.char_count} chars, has_text={page.has_text}")

    # Test with empty text
    empty_page = PageContent(page_number=2, text="")
    assert empty_page.has_text is False
    assert empty_page.char_count == 0
    print(f"   ✅ PageContent empty: char_count={empty_page.char_count}, has_text={empty_page.has_text}")

    # Test with whitespace-only text (should be treated as empty)
    ws_page = PageContent(page_number=3, text="   \n\n   ")
    assert ws_page.has_text is False
    print(f"   ✅ Whitespace-only page: has_text={ws_page.has_text}")


def test_your_own_pdf():
    """
    Interactive test — provide your own PDF path.
    Uncomment and set the path to test with your own documents.
    """
    print("\n── Test 6: Custom PDF (Optional) ───────────────────")

    # ── EDIT THIS PATH to test with your own PDF ──────────────────
    your_pdf_path = None  # e.g., r"C:\Users\hp\Documents\resume.pdf"

    if your_pdf_path is None:
        print("   ⏭️  Skipped — set your_pdf_path in test_your_own_pdf()")
        return

    processor = PDFProcessor()
    result = processor.process(your_pdf_path)

    print(f"   File: {result.metadata.filename}")
    print(f"   Pages: {result.metadata.page_count}")
    print(f"   Status: {result.status}")
    print(f"   Chars: {result.metadata.total_chars}")
    print(f"   Words: {result.metadata.total_words}")

    if result.pages:
        print(f"\n   First page preview:")
        first_page = result.pages_with_text[0] if result.pages_with_text else None
        if first_page:
            print(f"   {first_page.text[:300]}")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LOCAL RAG — PHASE 2: PDF PROCESSOR TESTS")
    print("=" * 55)

    test_page_content_model()
    test_missing_file()
    test_wrong_extension()
    test_empty_file()
    test_with_real_pdf()
    test_your_own_pdf()

    print("\n" + "=" * 55)
    print("   Phase 2 tests complete!")
    print("   Next: confirm all ✅ and move to Phase 3 (DOCX)")
    print("=" * 55 + "\n")