"""
config.py — Centralized Configuration Management
=================================================
Loads all settings from the .env file using Pydantic Settings.
Every module imports from here instead of reading env vars directly.
This gives us:
  - Type safety
  - Default values
  - Single source of truth
  - Easy testing (override settings in tests)
"""

from pydantic_settings import BaseSettings  # Pydantic v2 settings
from pydantic import Field                  # For field-level config
from pathlib import Path                    # OS-agnostic file paths
from functools import lru_cache             # Cache the settings object
import os


class Settings(BaseSettings):
    """
    All application settings.
    Pydantic automatically reads these from environment variables
    or the .env file. Variable names are case-insensitive.
    """

    # ── Application ──────────────────────────────────────────────
    app_name: str = Field(default="LocalRAG", description="Application name")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ── Paths ─────────────────────────────────────────────────────
    # Field(...) means required — no default value
    data_dir: str = Field(default="./data")
    documents_dir: str = Field(default="./data/documents")
    vector_store_dir: str = Field(default="./data/vector_store")
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/local_rag.db"
    )

    # ── Embedding Model ───────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name from HuggingFace"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Output vector size. Must match the model above."
    )

    # ── Text Chunking ─────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        description="Max characters per chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between consecutive chunks to preserve context"
    )

    # ── Retrieval ─────────────────────────────────────────────────
    top_k_results: int = Field(
        default=5,
        description="Number of similar chunks to retrieve per query"
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score (0-1). Lower = more results."
    )

    # ── Ollama ────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")
    ollama_timeout: int = Field(default=120, description="Seconds")

    # ── FAISS ─────────────────────────────────────────────────────
    faiss_index_file: str = Field(default="./data/vector_store/index.faiss")
    faiss_metadata_file: str = Field(
        default="./data/vector_store/metadata.json"
    )

    # ── File Upload ───────────────────────────────────────────────
    max_upload_size_mb: int = Field(
        default=50,
        description="Maximum file size in megabytes"
    )
    allowed_extensions: list[str] = Field(
        default=[".pdf", ".docx"],
        description="Allowed file extensions for upload"
    )

    def ensure_directories(self) -> None:
        """
        Create all required directories if they don't exist.
        Called once at application startup.
        """
        dirs = [
            self.data_dir,
            self.documents_dir,
            self.vector_store_dir,
        ]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)

    class Config:
        """
        Pydantic settings configuration.
        env_file tells it where to read environment variables from.
        """
        env_file = ".env"          # Load from .env file
        env_file_encoding = "utf-8"
        case_sensitive = False     # APP_NAME and app_name both work
        extra = "ignore"           # Ignore unknown env vars


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Why lru_cache?
    - Settings is read from disk. We don't want to re-read .env
      on every function call — that would be slow.
    - lru_cache ensures this function runs exactly once and returns
      the same object every time after that.

    Usage:
        from backend.config import get_settings
        settings = get_settings()
        print(settings.ollama_model)
    """
    settings = Settings()
    settings.ensure_directories()  # Create dirs on first load
    return settings


# ── Module-level convenience instance ──────────────────────────────
# Other modules can do: from backend.config import settings
settings = get_settings()