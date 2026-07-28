"""
processors/__init__.py
Exports document processors for clean imports elsewhere.
"""

from backend.processors.pdf_processor import PDFProcessor
from backend.processors.docx_processor import DOCXProcessor

__all__ = ["PDFProcessor", "DOCXProcessor"]