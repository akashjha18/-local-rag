"""
chunker.py — Recursive Text Chunking Engine
============================================
Splits ProcessedDocument text into overlapping chunks using
a recursive separator strategy.

Key design decisions:
  1. Recursive splitting: tries large separators first (paragraphs),
     falls back to smaller ones (sentences, words) only when needed.
     This preserves semantic units as much as possible.

  2. Overlap: consecutive chunks share `overlap` characters.
     This ensures context isn't lost at chunk boundaries —
     a sentence split across two chunks appears in both.

  3. Metadata attachment: every chunk knows its source document,
     page number, and character position. This is non-negotiable
     for a production RAG system.

  4. Page-aware: chunks respect [Page N] markers inserted by
     pdf_processor.py, so page attribution is accurate.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Optional

from backend.models.document_models import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    ProcessedDocument,
)
from backend.config import get_settings
from backend.utils.logger import logger


# ── Separator hierarchy ────────────────────────────────────────────
# Ordered from "best split point" to "last resort".
# The chunker tries each separator in order, splitting only when
# the resulting pieces are within the target size.
SEPARATORS = [
    "\n\n",   # Paragraph break      — ideal split point
    "\n",     # Line break           — second choice
    ". ",     # Sentence end         — third choice
    "! ",     # Exclamation          — fourth
    "? ",     # Question mark        — fifth
    "; ",     # Semicolon            — sixth
    ", ",     # Comma                — seventh
    " ",      # Word boundary        — eighth (last natural)
    "",       # Character boundary   — absolute last resort
]

# Minimum chunk size — chunks smaller than this are merged with next
MIN_CHUNK_SIZE = 50  # characters


class TextChunker:
    """
    Splits documents into overlapping text chunks.

    This is a clean-room implementation of recursive character
    text splitting — the same strategy used by LangChain's
    RecursiveCharacterTextSplitter, but with full source tracking.

    Args:
        chunk_size:    Target maximum characters per chunk
        chunk_overlap: Characters shared between consecutive chunks
        separators:    Ordered list of split points to try

    Usage:
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk_document(processed_doc, doc_id="abc123")
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[list[str]] = None,
    ):
        settings = get_settings()

        # Use provided values, or fall back to .env config
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or SEPARATORS

        # Validation: overlap must be less than chunk size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        logger.debug(
            f"TextChunker initialized: "
            f"size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def chunk_document(
        self,
        document: ProcessedDocument,
        document_id: str,
    ) -> list[DocumentChunk]:
        """
        Split a ProcessedDocument into overlapping DocumentChunks.

        This is the main public method. It:
          1. Extracts text from the document
          2. Builds a page-number lookup map
          3. Splits text recursively
          4. Attaches metadata to each chunk

        Args:
            document:    ProcessedDocument from pdf/docx processor
            document_id: Unique ID for this document in our system

        Returns:
            List of DocumentChunk, ordered as they appear in the doc.
            Empty list if document has no text.
        """
        if not document.full_text or not document.full_text.strip():
            logger.warning(
                f"Document {document.metadata.filename} has no text to chunk"
            )
            return []

        logger.info(
            f"Chunking: {document.metadata.filename} | "
            f"Total chars: {document.metadata.total_chars}"
        )

        # ── Build page position map ────────────────────────────────
        # The PDF processor inserted [Page N] markers in the text.
        # We build a sorted list of (char_position, page_number) pairs
        # so we can look up which page any character belongs to.
        page_map = self._build_page_map(document.full_text)

        # ── Split text into raw text pieces ───────────────────────
        raw_chunks = self._split_text(document.full_text)

        # ── Attach metadata to each chunk ─────────────────────────
        document_chunks = []
        char_cursor = 0   # Tracks our position in the full text

        for chunk_index, chunk_text in enumerate(raw_chunks):

            # Find where this chunk starts in the full text
            # (search from current cursor for efficiency)
            char_start = document.full_text.find(chunk_text, char_cursor)
            if char_start == -1:
                # Fallback: scan from beginning (handles edge cases)
                char_start = document.full_text.find(chunk_text)
            char_end = char_start + len(chunk_text)

            # Advance cursor (minus overlap so next chunk can find its start)
            char_cursor = max(0, char_end - self.chunk_overlap)

            # Determine page number from position
            page_number = self._get_page_number(char_start, page_map)

            # Remove [Page N] markers from the chunk text itself
            # (they're useful for position tracking, not for the LLM)
            clean_text = self._remove_page_markers(chunk_text)

            # Skip chunks that are too small after cleaning
            if len(clean_text.strip()) < MIN_CHUNK_SIZE:
                logger.debug(
                    f"Skipping tiny chunk {chunk_index}: "
                    f"{len(clean_text)} chars"
                )
                continue

            # Build unique chunk ID
            chunk_id = f"{document_id}_chunk_{chunk_index:04d}"

            # Build metadata
            metadata = ChunkMetadata(
                document_id=document_id,
                filename=document.metadata.filename,
                document_type=document.metadata.document_type.value,
                page_number=page_number,
                chunk_index=chunk_index,
                total_chunks=len(raw_chunks),  # Approximate (pre-filter)
                char_start=char_start,
                char_end=char_end,
            )

            document_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=clean_text.strip(),
                metadata=metadata,
            ))

        # Update total_chunks to reflect actual count (post-filter)
        for chunk in document_chunks:
            chunk.metadata.total_chunks = len(document_chunks)

        logger.info(
            f"Chunking complete: {document.metadata.filename} | "
            f"{len(document_chunks)} chunks | "
            f"avg {sum(c.word_count for c in document_chunks) // max(len(document_chunks), 1)} words/chunk"
        )

        return document_chunks

    # ── Core Splitting Logic ───────────────────────────────────────

    def _split_text(self, text: str) -> list[str]:
        """
        Recursively split text using the separator hierarchy.

        Algorithm:
          1. Try the first separator (paragraph break)
          2. Split the text on it
          3. For each piece:
             - If piece fits in chunk_size → keep it as-is
             - If piece is too large → recurse with next separator
          4. Merge small adjacent pieces to fill up to chunk_size
          5. Apply overlap when merging

        This "split then merge" approach is key:
        - Split creates natural-boundary pieces
        - Merge fills chunks efficiently without wasting space

        Args:
            text: Text to split (any length)

        Returns:
            List of text strings, each <= chunk_size characters
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """
        Core recursive splitting implementation.

        Args:
            text:       Text to split
            separators: Remaining separators to try (shrinks each recursion)

        Returns:
            List of chunks, each within chunk_size
        """
        # Base case: text already fits in one chunk
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Base case: no separators left — force-split by character
        if not separators:
            return self._hard_split(text)

        # Try the current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on this separator
        if separator == "":
            # Empty separator = split every character
            splits = list(text)
        elif separator in text:
            splits = text.split(separator)
        else:
            # This separator doesn't appear in text — try next one
            return self._recursive_split(text, remaining_separators)

        # Process each split piece
        good_splits = []   # Pieces that fit within chunk_size
        current = ""       # Accumulator for building chunks

        for split in splits:
            # Reattach the separator (we split ON it, losing it)
            piece = split + separator if separator != "" else split

            if len(current) + len(piece) <= self.chunk_size:
                # Piece fits with current accumulator — keep building
                current += piece
            else:
                if current:
                    good_splits.append(current)
                    current = ""

                if len(piece) <= self.chunk_size:
                    # Piece fits alone
                    current = piece
                else:
                    # Piece is too large — recurse with next separator
                    if current:
                        good_splits.append(current)
                        current = ""
                    sub_chunks = self._recursive_split(
                        piece, remaining_separators
                    )
                    good_splits.extend(sub_chunks)

        # Don't forget the last accumulator
        if current and current.strip():
            good_splits.append(current)

        # Apply overlap and merge small pieces
        return self._merge_with_overlap(good_splits)

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        """
        Merge small adjacent splits and apply overlap.

        This is the second half of the algorithm — after splitting,
        we merge pieces that are too small, and add overlap between
        consecutive chunks.

        The overlap works by including the last `chunk_overlap`
        characters of the previous chunk at the start of the next one.

        Args:
            splits: List of text pieces from splitting

        Returns:
            Final list of chunks with overlap applied
        """
        if not splits:
            return []

        chunks = []
        current_chunk = ""

        for split in splits:
            # Check if adding this split exceeds chunk size
            candidate = current_chunk + split

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Current chunk is full — finalize it
                if current_chunk.strip():
                    chunks.append(current_chunk)

                # Start new chunk WITH overlap from previous
                if chunks:
                    # Take the last `chunk_overlap` chars of previous chunk
                    overlap_text = chunks[-1][-self.chunk_overlap:]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split

                # If even with overlap it's too large, start fresh
                if len(current_chunk) > self.chunk_size:
                    current_chunk = split

        # Finalize last chunk
        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """
        Last resort: split by character count with overlap.
        Used when no separator works and text is too long.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            # Advance by chunk_size minus overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # ── Helper Methods ─────────────────────────────────────────────

    def _build_page_map(self, text: str) -> list[tuple[int, int]]:
        """
        Build a map of (character_position → page_number).

        Scans the full text for [Page N] markers inserted by
        pdf_processor._assemble_full_text().

        Returns:
            Sorted list of (char_position, page_number) tuples.
            e.g., [(0, 1), (1523, 2), (3041, 3)]

        For DOCX files which use section numbers instead, this
        still works because we format them the same way.
        """
        page_map = []

        # Pattern matches [Page 1], [Page 12], etc.
        pattern = re.compile(r"\[Page (\d+)\]")

        for match in pattern.finditer(text):
            page_number = int(match.group(1))
            char_position = match.start()
            page_map.append((char_position, page_number))

        if not page_map:
            # No page markers (e.g., plain DOCX without them)
            # Default everything to page 1
            page_map = [(0, 1)]

        return sorted(page_map)  # Sort by position

    def _get_page_number(
        self, char_position: int, page_map: list[tuple[int, int]]
    ) -> int:
        """
        Look up which page a character position belongs to.

        Uses binary search logic: finds the last page marker
        that appears BEFORE the given character position.

        Args:
            char_position: Character index in full_text
            page_map:      Sorted list from _build_page_map()

        Returns:
            Page number (1-indexed)
        """
        page_number = 1  # Default

        for pos, page in page_map:
            if pos <= char_position:
                page_number = page
            else:
                break  # Page markers are sorted — stop when we pass the position

        return page_number

    def _remove_page_markers(self, text: str) -> str:
        """
        Remove [Page N] markers from chunk text.

        These markers are for internal position tracking only.
        The LLM doesn't need to see them — it would just confuse it.
        """
        return re.sub(r"\[Page \d+\]\n?", "", text)

    def get_chunk_stats(self, chunks: list[DocumentChunk]) -> dict:
        """
        Compute statistics about a set of chunks.
        Useful for logging and debugging chunking quality.

        Returns:
            Dict with min/max/avg chunk sizes and word counts
        """
        if not chunks:
            return {}

        char_counts = [len(c.text) for c in chunks]
        word_counts = [c.word_count for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chars": sum(char_counts) // len(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_words": sum(word_counts) // len(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
        }