# 🔍 Local RAG System

A production-quality **Retrieval Augmented Generation (RAG)** system that runs completely offline. Upload PDFs and DOCX files, ask questions in natural language, and get accurate AI-generated answers with source citations.

## ✨ Features

- 📄 **PDF & DOCX support** — extract and index any document
- 🧠 **Local embeddings** — Sentence Transformers (all-MiniLM-L6-v2)
- ⚡ **FAISS vector search** — millisecond similarity search
- 🤖 **Local LLM** — Mistral / Phi-3 / Llama 3 via Ollama
- 🔒 **100% offline** — your data never leaves your machine
- 📌 **Source citations** — every answer shows which page it came from
- 🗑️ **Document management** — upload, list, delete documents
- 🏗️ **Clean architecture** — FastAPI backend, React frontend

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI |
| Embeddings | Sentence Transformers |
| Vector DB | FAISS |
| LLM | Ollama (Mistral / Phi-3) |
| Document Processing | PyPDF, python-docx |
| Database | SQLite |
| Frontend | React + Vite + Tailwind CSS |
| Deployment | Docker |

## 🏛️ Architecture
User Query
│
▼
FastAPI Backend
│
├── PDFProcessor / DOCXProcessor → Extract text
├── TextChunker → Split into chunks
├── EmbeddingService → Generate vectors
├── FAISSVectorStore → Store & search vectors
├── Retriever → Find relevant chunks
└── OllamaLLM → Generate answer

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/local-rag.git
cd local-rag

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Ollama

```bash
# Pull a local LLM (choose one)
ollama pull phi3:mini     # Recommended for 4GB VRAM
ollama pull mistral       # Better quality, needs 6GB+ VRAM
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and set OLLAMA_MODEL=phi3:mini (or mistral)
```

### Run Tests

```bash
python tests/test_phase9.py
```

## 📁 Project Structure

local-rag/
│
├── backend/
│ ├── api/ # FastAPI routes
│ ├── rag/ # Core RAG engine
│ │ ├── embeddings.py # Sentence Transformers
│ │ ├── vector_store.py # FAISS operations
│ │ ├── retriever.py # Similarity search
│ │ ├── llm.py # Ollama integration
│ │ └── pipeline.py # Master orchestrator
│ ├── processors/ # PDF & DOCX extraction
│ ├── models/ # Data models
│ └── config.py # Configuration
│
├── frontend/ # React + Vite + Tailwind
├── data/ # Documents & vector store
├── tests/ # Phase-by-phase tests
└── docker/ # Docker deployment

## 🔧 Configuration

All settings in `.env`:

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
OLLAMA_MODEL=phi3:mini
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

## 📊 Performance

| Operation | Time (CPU) |
|-----------|------------|
| PDF ingestion (10 pages) | ~2s |
| Embedding 57 chunks | ~1.7s |
| FAISS search | ~30ms |
| LLM generation (phi3:mini) | ~15-60s |

## 🗺️ Roadmap

- [x] PDF extraction
- [x] DOCX extraction
- [x] Text chunking with overlap
- [x] Local embeddings
- [x] FAISS vector store
- [x] Ollama LLM integration
- [x] RAG pipeline
- [ ] FastAPI REST API
- [ ] React frontend
- [ ] Docker deployment
- [ ] Hybrid search (BM25 + Vector)
- [ ] Streaming responses
- [ ] Multi-user support

## 👨‍💻 Author

**Akash Jha**
MCA — AI & ML | JECRC University, Jaipur
[GitHub](https://github.com/akashjha18)

## 📄 License

MIT License — feel free to use for learning and projects.