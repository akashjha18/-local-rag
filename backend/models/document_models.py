"""
document_models.py — Shared Data Models for Document Processing
===============================================================
These are plain Python dataclasses (not Pydantic) used internally
within the backend pipeline. They are NOT API models — those live
in request_models.py and response_models.py.

Why dataclasses here instead of Pydantic?
- Pydantic is for API boundary validation (JSON in/out)
- Dataclasses are lightweight for internal data passing
- No serialization overhead within the Python process
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DocumentType(str, Enum):
    """
    Supported document types.
    str Enum means the value IS the string — useful for storage.
    """
    PDF = "pdf"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """
    Tracks what happened during processing.
    Returned to the caller so they know exactly what occurred.
    """
    SUCCESS = "success"
    PARTIAL = "partial"          # Some pages failed, others worked
    ENCRYPTED = "encrypted"      # Password-protected file
    EMPTY = "empty"              # File had no extractable text
    FAILED = "failed"            # Complete failure


@dataclass
class PageContent:
    """
    Text content of a single page.

    Attributes:
        page_number: 1-indexed page number (human-readable)
        text:        Extracted text from this page
        char_count:  Length of text for quick stats
        has_text:    False if page was an image/blank
    """
    page_number: int
    text: str
    char_count: int = field(init=False)  # Computed automatically
    has_text: bool = field(init=False)   # Computed automatically

    def __post_init__(self):
        """
        Called automatically after __init__.
        We compute derived fields here so callers don't have to.
        """
        # Strip leading/trailing whitespace before counting
        self.text = self.text.strip()
        self.char_count = len(self.text)
        self.has_text = self.char_count > 0


@dataclass
class DocumentMetadata:
    """
    File-level metadata extracted from the document.
    Optional fields because not all PDFs have them set.
    """
    filename: str
    file_size_bytes: int
    document_type: DocumentType
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None   # Software that created the PDF
    total_chars: int = 0
    total_words: int = 0


@dataclass
class ProcessedDocument:
    """
    The complete output of processing a document.
    This is what pdf_processor.py and docx_processor.py return.

    Attributes:
        metadata:         File-level info
        pages:            List of per-page content
        full_text:        All pages joined into one string
        status:           SUCCESS, PARTIAL, ENCRYPTED, etc.
        error_message:    Set if status != SUCCESS
        processing_time:  Seconds taken to process
    """
    metadata: DocumentMetadata
    pages: list[PageContent]
    full_text: str
    status: ProcessingStatus
    error_message: Optional[str] = None
    processing_time: float = 0.0

    @property
    def is_successful(self) -> bool:
        """Convenience check for callers."""
        return self.status in (ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL)

    @property
    def pages_with_text(self) -> list[PageContent]:
        """Return only pages that had extractable text."""
        return [p for p in self.pages if p.has_text]


# ── ADD THESE to the bottom of document_models.py ─────────────────


@dataclass
class ChunkMetadata:
    """
    Source tracking information attached to every chunk.

    This is what powers "Show Sources" in the UI.
    Every chunk carries its provenance so we can always
    tell the user exactly which document and page an answer came from.
    """
    document_id: str          # Unique ID of the source document
    filename: str             # Original filename (e.g., "report.pdf")
    document_type: str        # "pdf" or "docx"
    page_number: int          # Which page this chunk came from
    chunk_index: int          # Position of this chunk in the document
    total_chunks: int         # Total chunks in this document
    char_start: int           # Start character position in full_text
    char_end: int             # End character position in full_text


@dataclass
class DocumentChunk:
    """
    A single chunk of text ready for embedding.

    This is the fundamental unit of the RAG system.
    Everything downstream (embeddings, FAISS, retrieval) works
    with DocumentChunk objects.

    Attributes:
        chunk_id:    Globally unique ID for this chunk
        text:        The actual text content to embed and search
        metadata:    Source tracking (file, page, position)
        word_count:  Quick stat for filtering tiny/huge chunks
    """
    chunk_id: str             # e.g., "doc123_chunk_0042"
    text: str
    metadata: ChunkMetadata
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())

    def __repr__(self) -> str:
        return (
            f"DocumentChunk(id={self.chunk_id}, "
            f"words={self.word_count}, "
            f"page={self.metadata.page_number})"
        )

# ── ADD at the bottom of document_models.py ───────────────────────

@dataclass
class SearchResult:
    """
    A single result from FAISS similarity search.

    Returned by FAISSVectorStore.search().
    Contains everything the RAG pipeline needs to:
      1. Pass the chunk text to the LLM as context
      2. Show the user which document/page it came from
      3. Display a confidence score

    Attributes:
        chunk_id:         Unique chunk identifier
        text:             The actual chunk text (LLM context)
        score:            Cosine similarity score (0.0 to 1.0)
        filename:         Source document name
        document_id:      Source document ID
        page_number:      Page this chunk came from
        chunk_index:      Position of chunk within document
        document_type:    "pdf" or "docx"
    """
    chunk_id: str
    text: str
    score: float
    filename: str
    document_id: str
    page_number: int
    chunk_index: int
    document_type: str

    @property
    def source_label(self) -> str:
        """Human-readable source string for UI display."""
        return f"{self.filename} (Page {self.page_number})"

    @property
    def confidence_percent(self) -> int:
        """Score as integer percentage for display."""
        return min(100, int(self.score * 100))

# ── ADD at the bottom of document_models.py ───────────────────────

@dataclass
class LLMResponse:
    """
    Complete response from the LLM service.

    Returned by OllamaLLM.generate_answer().
    Contains everything needed to render the final answer in the UI.

    Attributes:
        answer:          The generated text answer
        model:           Which Ollama model produced it
        sources:         SearchResult list used as context
        prompt_tokens:   Tokens in the prompt (for monitoring)
        completion_tokens: Tokens in the answer
        total_tokens:    prompt + completion
        generation_time: Seconds taken by Ollama
        success:         False if LLM call failed
        error:           Error message if success=False
    """
    answer: str
    model: str
    sources: list            # list[SearchResult] — avoid circular import
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def unique_sources(self) -> list:
        """Deduplicated sources by filename+page for UI display."""
        seen = set()
        unique = []
        for source in self.sources:
            key = f"{source.filename}:{source.page_number}"
            if key not in seen:
                seen.add(key)
                unique.append(source)
        return unique


# ── ADD at the bottom of document_models.py ───────────────────────

@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.
    This is what the FastAPI endpoint returns to the frontend.

    Attributes:
        query:           The original user question
        answer:          LLM-generated answer
        sources:         Unique source documents cited
        model:           LLM model used
        retrieval_time:  Seconds for FAISS search
        generation_time: Seconds for LLM generation
        total_time:      End-to-end seconds
        chunks_retrieved: How many chunks were found
        success:         False if pipeline failed
        error:           Error message if success=False
    """
    query: str
    answer: str
    sources: list          # list[SearchResult]
    model: str
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
    success: bool = True
    error: Optional[str] = None


@dataclass
class IngestResult:
    """
    Result of ingesting a document into the RAG system.
    Returned by RAGPipeline.ingest().

    Attributes:
        document_id:     Unique ID assigned to this document
        filename:        Original filename
        chunks_indexed:  Number of chunks stored in FAISS
        processing_time: Total seconds for ingest
        status:          "success" | "failed" | "empty"
        error:           Error message if status != "success"
        page_count:      Pages in the document
        word_count:      Words extracted
    """
    document_id: str
    filename: str
    chunks_indexed: int
    processing_time: float
    status: str
    error: Optional[str] = None
    page_count: int = 0
    word_count: int = 0
    