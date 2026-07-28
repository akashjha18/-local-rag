"""
pdf_processor.py — PDF Text Extraction Engine
=============================================
Handles all PDF-related text extraction using PyPDF.

Responsibilities:
  1. Validate the PDF file before processing
  2. Extract text from each page individually
  3. Clean and normalize the extracted text
  4. Return a structured ProcessedDocument with metadata
  5. Handle errors gracefully (encrypted, corrupted, empty)

Design principle:
  This module is PURE — it only reads files, never writes.
  It has no knowledge of FastAPI, databases, or embeddings.
  It does ONE thing: PDF → text.
"""

import time                          # For measuring processing duration
from pathlib import Path             # OS-agnostic file paths
from typing import Optional

import pypdf                         # Main PDF library
from pypdf import PdfReader          # The PDF reading class
from pypdf.errors import PdfReadError  # PyPDF-specific exceptions

from backend.models.document_models import (
    DocumentMetadata,
    DocumentType,
    PageContent,
    ProcessedDocument,
    ProcessingStatus,
)
from backend.utils.logger import logger


# ── Module-level constants ─────────────────────────────────────────
# Maximum file size: 100MB. Larger files risk memory issues.
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Minimum text per page to consider it "has text"
# (a page with just 5 chars is probably a blank or header)
MIN_PAGE_TEXT_LENGTH = 10


class PDFProcessor:
    """
    Extracts text from PDF files using PyPDF.

    Usage:
        processor = PDFProcessor()
        result = processor.process("path/to/file.pdf")
        print(result.full_text)
        print(result.metadata.page_count)
    """

    def __init__(self):
        """
        PDFProcessor has no state — all configuration comes through
        method parameters. This makes it safe to use as a singleton.
        """
        logger.debug("PDFProcessor initialized")

    def process(self, file_path: str | Path) -> ProcessedDocument:
        """
        Main entry point. Process a PDF file end-to-end.

        Args:
            file_path: Path to the PDF file on disk

        Returns:
            ProcessedDocument with all extracted content and metadata.
            NEVER raises an exception — errors are captured in the
            returned object's status and error_message fields.

        This "never raise" design pattern is important for a web API:
        we want to always return a response to the client, even if
        processing failed, rather than crashing the server.
        """
        start_time = time.time()                  # Start the clock
        file_path = Path(file_path)               # Ensure it's a Path object

        logger.info(f"Starting PDF processing: {file_path.name}")

        # ── Step 1: Validate the file ──────────────────────────────
        # This returns an error ProcessedDocument if validation fails,
        # or None if everything is OK.
        validation_error = self._validate_file(file_path)
        if validation_error:
            # Attach timing info even to error responses
            validation_error.processing_time = time.time() - start_time
            return validation_error

        # ── Step 2: Build metadata object ─────────────────────────
        # We do this before extraction so we can include it in errors
        file_size = file_path.stat().st_size
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_size_bytes=file_size,
            document_type=DocumentType.PDF,
            page_count=0,           # Will be updated after opening
        )

        # ── Step 3: Open and read the PDF ─────────────────────────
        try:
            reader = PdfReader(str(file_path))

            # ── Handle encrypted PDFs ──────────────────────────────
            # PyPDF sets .is_encrypted = True for password-protected files
            if reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {file_path.name}")
                return ProcessedDocument(
                    metadata=metadata,
                    pages=[],
                    full_text="",
                    status=ProcessingStatus.ENCRYPTED,
                    error_message=(
                        "This PDF is password-protected. "
                        "Please provide an unencrypted version."
                    ),
                    processing_time=time.time() - start_time,
                )

            # ── Update metadata with real page count ───────────────
            metadata.page_count = len(reader.pages)

            # ── Extract PDF document info (title, author, etc.) ────
            self._extract_pdf_metadata(reader, metadata)

            logger.debug(
                f"PDF opened: {metadata.page_count} pages, "
                f"Title: {metadata.title or 'untitled'}"
            )

        except PdfReadError as e:
            # PyPDF-specific error — corrupted or invalid PDF
            logger.error(f"PyPDF failed to read {file_path.name}: {e}")
            return ProcessedDocument(
                metadata=metadata,
                pages=[],
                full_text="",
                status=ProcessingStatus.FAILED,
                error_message=f"Could not read PDF file. It may be corrupted: {str(e)}",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error opening PDF {file_path.name}: {e}")
            return ProcessedDocument(
                metadata=metadata,
                pages=[],
                full_text="",
                status=ProcessingStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}",
                processing_time=time.time() - start_time,
            )

        # ── Step 4: Extract text from each page ───────────────────
        pages = self._extract_pages(reader, file_path.name)

        # ── Step 5: Assemble full text ─────────────────────────────
        # Join all page texts with a clear separator.
        # The separator helps the chunker know where page boundaries are.
        full_text = self._assemble_full_text(pages)

        # ── Step 6: Update metadata statistics ────────────────────
        metadata.total_chars = len(full_text)
        metadata.total_words = len(full_text.split())

        # ── Step 7: Determine final status ────────────────────────
        pages_with_text = [p for p in pages if p.has_text]
        status, error_message = self._determine_status(
            pages, pages_with_text, file_path.name
        )

        processing_time = time.time() - start_time

        logger.info(
            f"PDF processed: {file_path.name} | "
            f"Status: {status} | "
            f"Pages: {metadata.page_count} | "
            f"Text pages: {len(pages_with_text)} | "
            f"Chars: {metadata.total_chars} | "
            f"Time: {processing_time:.2f}s"
        )

        return ProcessedDocument(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            status=status,
            error_message=error_message,
            processing_time=processing_time,
        )

    # ── Private Methods ────────────────────────────────────────────

    def _validate_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        Validate the file before attempting to process it.

        Returns:
            None if validation passes (file is OK to process)
            ProcessedDocument with error status if validation fails

        Why validate first?
        - Fail fast: catch obvious errors without loading the whole file
        - Better error messages for common problems
        - Security: prevent processing of unexpected file types
        """

        # ── Check 1: File must exist ───────────────────────────────
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return self._make_error_doc(
                filename=file_path.name,
                status=ProcessingStatus.FAILED,
                error=f"File not found: {file_path}",
            )

        # ── Check 2: Must be a file, not a directory ───────────────
        if not file_path.is_file():
            return self._make_error_doc(
                filename=file_path.name,
                status=ProcessingStatus.FAILED,
                error=f"Path is not a file: {file_path}",
            )

        # ── Check 3: File size limit ───────────────────────────────
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return self._make_error_doc(
                filename=file_path.name,
                status=ProcessingStatus.FAILED,
                error=(
                    f"File too large: {size_mb:.1f}MB. "
                    f"Maximum allowed: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB"
                ),
            )

        # ── Check 4: Empty file ────────────────────────────────────
        if file_size == 0:
            return self._make_error_doc(
                filename=file_path.name,
                status=ProcessingStatus.FAILED,
                error="File is empty (0 bytes)",
            )

        # ── Check 5: PDF extension ─────────────────────────────────
        if file_path.suffix.lower() != ".pdf":
            return self._make_error_doc(
                filename=file_path.name,
                status=ProcessingStatus.FAILED,
                error=f"Not a PDF file. Extension: {file_path.suffix}",
            )

        return None  # All checks passed

    def _extract_pdf_metadata(
        self, reader: PdfReader, metadata: DocumentMetadata
    ) -> None:
        """
        Extract document metadata from PDF's built-in info dictionary.

        PDF files store optional metadata in a /Info dictionary.
        Not all PDFs have this, so we handle missing values gracefully.
        Mutates the metadata object in-place.
        """
        try:
            # reader.metadata is a dict-like object with PDF info
            # It can be None if the PDF has no info dictionary
            pdf_info = reader.metadata

            if pdf_info:
                # .get() returns None if key doesn't exist
                # We strip whitespace and convert None to None (not "None")
                raw_title = pdf_info.get("/Title", "")
                raw_author = pdf_info.get("/Author", "")
                raw_subject = pdf_info.get("/Subject", "")
                raw_creator = pdf_info.get("/Creator", "")

                # Only set if non-empty after stripping
                metadata.title = raw_title.strip() if raw_title else None
                metadata.author = raw_author.strip() if raw_author else None
                metadata.subject = raw_subject.strip() if raw_subject else None
                metadata.creator = raw_creator.strip() if raw_creator else None

        except Exception as e:
            # Metadata extraction failure is non-fatal
            # We log it but continue with the document
            logger.warning(f"Could not extract PDF metadata: {e}")

    def _extract_pages(
        self, reader: PdfReader, filename: str
    ) -> list[PageContent]:
        """
        Extract text from each page of the PDF.

        Returns a list of PageContent objects, one per page.
        Pages that fail extraction are included as empty pages
        (not skipped) so page numbering stays accurate.

        Args:
            reader:   PyPDF PdfReader (already opened)
            filename: Just for logging

        Returns:
            List of PageContent, length == number of pages
        """
        pages = []
        total_pages = len(reader.pages)

        for page_index in range(total_pages):
            page_number = page_index + 1  # Convert to 1-indexed

            try:
                # Get the PyPDF page object
                pdf_page = reader.pages[page_index]

                # extract_text() returns a string (may be empty for image pages)
                raw_text = pdf_page.extract_text()

                # raw_text can be None in some edge cases
                if raw_text is None:
                    raw_text = ""

                # Clean the extracted text
                cleaned_text = self._clean_text(raw_text)

                page_content = PageContent(
                    page_number=page_number,
                    text=cleaned_text,
                )

                # Log a warning for image-only pages
                if not page_content.has_text:
                    logger.debug(
                        f"{filename} — Page {page_number}/{total_pages}: "
                        f"no text extracted (possibly image-only)"
                    )
                else:
                    logger.debug(
                        f"{filename} — Page {page_number}/{total_pages}: "
                        f"{page_content.char_count} chars"
                    )

                pages.append(page_content)

            except Exception as e:
                # Page extraction failure — add empty page, continue
                logger.warning(
                    f"{filename} — Page {page_number}: extraction failed: {e}"
                )
                pages.append(PageContent(page_number=page_number, text=""))

        return pages

    def _clean_text(self, raw_text: str) -> str:
        """
        Clean and normalize extracted PDF text.

        PDF text often has issues:
        - Multiple consecutive spaces
        - Lines ending with hyphen-linebreak (word-wrap artifacts)
        - Excessive newlines
        - Null bytes and control characters

        This function fixes all of these for cleaner chunking later.

        Args:
            raw_text: Raw string from PyPDF's extract_text()

        Returns:
            Cleaned string
        """
        if not raw_text:
            return ""

        import re  # Regular expressions for text cleaning

        text = raw_text

        # ── Fix 1: Remove null bytes and control characters ────────
        # PDF sometimes includes \x00 (null) and other control chars
        text = text.replace("\x00", "")           # Remove null bytes
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # ── Fix 2: Fix hyphenated line breaks ─────────────────────
        # PDFs often hyphenate words across lines: "connec-\ntion" → "connection"
        text = re.sub(r"-\n(\w)", r"\1", text)

        # ── Fix 3: Normalize line endings ─────────────────────────
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # ── Fix 4: Collapse multiple spaces ───────────────────────
        # "hello    world" → "hello world"
        text = re.sub(r" {2,}", " ", text)

        # ── Fix 5: Collapse excessive blank lines ─────────────────
        # More than 2 consecutive newlines → exactly 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # ── Fix 6: Strip each line ─────────────────────────────────
        # Remove trailing spaces from each line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _assemble_full_text(self, pages: list[PageContent]) -> str:
        """
        Join all page texts into a single string.

        We use a clear page separator so downstream chunking
        can respect page boundaries when splitting.

        Args:
            pages: List of PageContent (may have empty pages)

        Returns:
            Single string with all page text
        """
        text_parts = []

        for page in pages:
            if page.has_text:
                # Include page marker for source tracking later
                # The chunker will use this to know which page a chunk came from
                text_parts.append(
                    f"[Page {page.page_number}]\n{page.text}"
                )

        # Join pages with double newline separator
        return "\n\n".join(text_parts)

    def _determine_status(
        self,
        all_pages: list[PageContent],
        pages_with_text: list[PageContent],
        filename: str,
    ) -> tuple[ProcessingStatus, Optional[str]]:
        """
        Determine the final processing status based on results.

        Returns:
            Tuple of (status, error_message)
        """
        total = len(all_pages)
        extracted = len(pages_with_text)

        if total == 0:
            return (
                ProcessingStatus.FAILED,
                "PDF has no pages",
            )

        if extracted == 0:
            # No pages had text — likely a scanned image PDF
            return (
                ProcessingStatus.EMPTY,
                (
                    f"No text could be extracted from {filename}. "
                    "This PDF may contain only scanned images. "
                    "OCR support is planned for a future phase."
                ),
            )

        if extracted < total:
            # Some pages worked, some didn't
            logger.warning(
                f"{filename}: {extracted}/{total} pages had extractable text"
            )
            return (
                ProcessingStatus.PARTIAL,
                f"Extracted text from {extracted} of {total} pages. "
                f"{total - extracted} pages may be image-only.",
            )

        # All pages extracted successfully
        return ProcessingStatus.SUCCESS, None

    def _make_error_doc(
        self,
        filename: str,
        status: ProcessingStatus,
        error: str,
    ) -> ProcessedDocument:
        """
        Helper to create a consistent error ProcessedDocument.
        Used in _validate_file() to avoid repetitive code.
        """
        logger.error(f"PDF validation failed for {filename}: {error}")
        return ProcessedDocument(
            metadata=DocumentMetadata(
                filename=filename,
                file_size_bytes=0,
                document_type=DocumentType.PDF,
                page_count=0,
            ),
            pages=[],
            full_text="",
            status=status,
            error_message=error,
        )