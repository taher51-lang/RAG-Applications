# ⚖️ NyayaSetu — Indian Legal Research Assistant

**NyayaSetu** is an AI-powered legal research tool that detects conflicts between Supreme Court of India judgments. It uses Retrieval-Augmented Generation (RAG) to analyze landmark and citing judgments, classify their relationship (AFFIRMS / NARROWS / OVERRULES), and present a structured legal analysis.

> **Current coverage:** Landmark Supreme Court judgments on employment & labor law.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                   React Frontend                     │
│          (Single-file SPA · Tailwind CSS)            │
├──────────────────────────────────────────────────────┤
│                   FastAPI Backend                     │
│            POST /query · POST /ingest · GET /health  │
├──────────────────────────────────────────────────────┤
│                  NyayaSetu Pipeline                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  Document    │  │  NyayaSetu   │  │  Conflict   │ │
│  │  Ingester    │  │  Retriever   │  │  Detector   │ │
│  └─────────────┘  └──────────────┘  └─────────────┘ │
├──────────────────────────────────────────────────────┤
│  ChromaDB (Vector Store) · HuggingFace Embeddings    │
│  BM25 (Keyword Search) · Cross-Encoder Re-ranker     │
│  Google Gemini 2.5 Flash (LLM)                       │
└──────────────────────────────────────────────────────┘
```

### RAG Pipeline

| Stage | Component | Detail |
|-------|-----------|--------|
| **Ingestion** | `DocumentIngester` | Parses PDFs via pdfplumber, extracts metadata, chunks with parent/child splitters, stores in ChromaDB |
| **Retrieval** | `NyayaSetuRetriever` | Hybrid search (BM25 + vector, 60/40 weighting) → Cross-Encoder re-ranking (BAAI/bge-reranker-base) |
| **Conflict Detection** | `ConflictDetector` | Gemini 2.5 Flash classifies the relationship and generates a structured legal analysis |
| **Orchestration** | `NyayaSetu` | Singleton that wires everything together via dependency injection |

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
cd FinRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the `FinRAG/` directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Ingest judgments (first run only)

```bash
python main.py
```

This ingests the bundled PDFs into the ChromaDB vector store.

### 4. Start the API server

```bash
python api.py
```

The server starts at **http://localhost:8000** — the frontend is served automatically.

---

## 📡 API Endpoints

### `GET /health`
Returns `{"status": "ok"}` when the pipeline is initialized.

### `POST /query`
```json
// Request
{ "query": "Can contract workers claim permanent employment?" }

// Response
{
  "answer": "...",
  "conflict_type": "NARROWS",
  "landmark_case": "State Of Haryana vs Piara Singh, 1992",
  "citing_case": "Secretary, State Of Karnataka vs Umadevi, 2006",
  "conflict_detected": true
}
```

### `POST /ingest`
```json
{
  "pdf_path": "/absolute/path/to/judgment.pdf",
  "doc_type": "landmark",
  "doc_id": "unique_id",
  "parent_id": null
}
```

---

## 🖥️ Frontend Features

- **Dark glassmorphism UI** with ambient animations
- **Chat interface** for querying the RAG pipeline
- **Conflict badges** — color-coded (🔴 OVERRULES · 🟠 NARROWS · 🟢 AFFIRMS)
- **Case cards** showing landmark and citing judgments side by side
- **Legal Evolution Timeline** — horizontal visual showing how the law evolved between judgments
- **Sample query chips** for quick exploration

---

## 📁 Project Structure

```
FinRAG/
├── main.py              # NyayaSetu RAG pipeline (core logic)
├── api.py               # FastAPI backend
├── .env                 # Environment variables (not committed)
├── frontend/
│   └── index.html       # React + Tailwind single-file SPA
├── nyayasetu_db/        # ChromaDB persistent storage
├── *.PDF                # Supreme Court judgment PDFs
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace) |
| **Re-ranker** | BAAI/bge-reranker-base (Cross-Encoder) |
| **Vector DB** | ChromaDB (persistent) |
| **Keyword Search** | BM25Retriever |
| **Framework** | LangChain (Classic + Community) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Tailwind CSS (CDN) |

---

## ⚠️ Disclaimer

NyayaSetu is a research tool and **not a substitute for professional legal advice**. Always consult a qualified legal professional for legal matters.
