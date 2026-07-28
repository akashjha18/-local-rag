"""
test_phase1.py — Phase 1 Verification
======================================
Run this to confirm your environment is set up correctly.
"""

import sys
import os


def test_python_version():
    """Python 3.10+ required for modern type hints."""
    version = sys.version_info
    assert version.major == 3 and version.minor >= 10, (
        f"Python 3.10+ required. You have {version.major}.{version.minor}"
    )
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def test_imports():
    """Test that all critical packages are installed."""
    packages = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "sqlalchemy": "SQLAlchemy",
        "pypdf": "PyPDF",
        "docx2txt": "Docx2txt",
        "sentence_transformers": "Sentence Transformers",
        "faiss": "FAISS",
        "numpy": "NumPy",
        "ollama": "Ollama",
        "pydantic": "Pydantic",
        "loguru": "Loguru",
    }

    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name} — MISSING: {e}")


def test_config():
    """Test that config loads correctly from .env."""
    # Add backend to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from backend.config import get_settings

    settings = get_settings()
    print(f"✅ Config loaded — App: {settings.app_name} v{settings.app_version}")
    print(f"   Model: {settings.embedding_model}")
    print(f"   Ollama: {settings.ollama_base_url}")


def test_directories():
    """Test that all required directories exist."""
    from pathlib import Path
    dirs = [
        "data",
        "data/documents",
        "data/vector_store",
        "backend",
        "backend/api",
        "backend/rag",
        "backend/processors",
        "backend/services",
        "backend/database",
    ]
    for d in dirs:
        if Path(d).exists():
            print(f"✅ Directory: {d}")
        else:
            print(f"❌ Missing directory: {d}")


def test_logger():
    """Test logger setup."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from backend.utils.logger import logger, setup_logger
    setup_logger(debug=True)
    logger.info("Logger test successful")
    print("✅ Logger working")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  LOCAL RAG — PHASE 1 VERIFICATION")
    print("="*50 + "\n")

    print("── Python Version ──────────────────")
    test_python_version()

    print("\n── Package Imports ─────────────────")
    test_imports()

    print("\n── Configuration ───────────────────")
    test_config()

    print("\n── Directory Structure ─────────────")
    test_directories()

    print("\n── Logger ──────────────────────────")
    test_logger()

    print("\n" + "="*50)
    print("  Phase 1 complete. Report back!")
    print("="*50 + "\n")