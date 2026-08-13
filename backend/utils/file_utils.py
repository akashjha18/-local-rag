"""
file_utils.py — File Utility Functions
========================================
Helpers for file validation, hashing, and management.
Used by the document upload pipeline.
"""

import hashlib
from pathlib import Path
from typing import Optional

from backend.utils.logger import logger


def compute_file_hash(file_path: str | Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Used for duplicate detection — if two files have the same hash,
    they are identical regardless of filename.

    Args:
        file_path: Path to the file

    Returns:
        Hex string of SHA-256 hash (64 characters)

    Example:
        hash1 = compute_file_hash("report.pdf")
        hash2 = compute_file_hash("report_copy.pdf")
        if hash1 == hash2:
            print("Same file!")
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in 8KB chunks — handles large files without
        # loading the entire file into memory
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_content_hash(content: bytes) -> str:
    """
    Compute SHA-256 hash of raw bytes (file content).
    Used when the file is already in memory (upload handler).

    Args:
        content: Raw file bytes

    Returns:
        Hex string of SHA-256 hash
    """
    return hashlib.sha256(content).hexdigest()


def get_file_size_mb(file_path: str | Path) -> float:
    """Return file size in megabytes."""
    return Path(file_path).stat().st_size / (1024 * 1024)


def is_allowed_extension(filename: str, allowed: set[str]) -> bool:
    """
    Check if a filename has an allowed extension.

    Args:
        filename: The filename to check (e.g., "report.pdf")
        allowed:  Set of allowed extensions (e.g., {".pdf", ".docx"})

    Returns:
        True if extension is allowed
    """
    suffix = Path(filename).suffix.lower()
    return suffix in allowed


def sanitize_filename(filename: str) -> str:
    """
    Remove unsafe characters from a filename.

    Prevents path traversal attacks and filesystem issues.
    Keeps only alphanumeric, dots, hyphens, underscores, spaces.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for storage
    """
    import re

    # Step 1: Take only the basename — strips any directory components
    # "../../../etc/passwd" → "passwd"
    # "folder/file.pdf"     → "file.pdf"
    filename = Path(filename).name

    # Step 2: Remove or replace unsafe characters
    # Keep only: letters, digits, dots, hyphens, underscores, spaces
    safe = re.sub(r'[^\w\s\-.]', '_', filename)

    # Step 3: Remove consecutive dots (prevents "file..pdf" tricks)
    safe = re.sub(r'\.{2,}', '.', safe)

    # Step 4: Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)

    # Step 5: Remove leading/trailing dots, spaces, underscores
    safe = safe.strip('. _')

    # Step 6: Fallback if filename becomes empty after sanitization
    return safe if safe else "document"


def find_file_by_document_id(
    docs_dir: str | Path,
    document_id: str,
) -> Optional[Path]:
    """
    Find a file in the documents directory by its document ID prefix.

    Files are stored as: "{document_id}_{original_filename}"
    This function finds the file given just the document_id.

    Args:
        docs_dir:    Directory to search
        document_id: UUID prefix to look for

    Returns:
        Path to the file, or None if not found
    """
    docs_dir = Path(docs_dir)
    matches = list(docs_dir.glob(f"{document_id}_*"))

    if not matches:
        logger.warning(f"No file found for document_id: {document_id}")
        return None

    if len(matches) > 1:
        logger.warning(
            f"Multiple files found for document_id {document_id}: {matches}"
        )

    return matches[0]


def extract_original_filename(stored_filename: str, document_id: str) -> str:
    """
    Extract the original filename from a stored UUID-prefixed filename.

    Stored as: "{document_id}_{original_filename}"
    Returns:   "{original_filename}"

    Args:
        stored_filename: e.g., "abc123-uuid_report.pdf"
        document_id:     e.g., "abc123-uuid"

    Returns:
        "report.pdf"
    """
    prefix = f"{document_id}_"
    if stored_filename.startswith(prefix):
        return stored_filename[len(prefix):]
    return stored_filename