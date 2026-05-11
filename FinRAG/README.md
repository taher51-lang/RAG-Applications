# ⚖️ NyayaSetu — Indian Legal Research Engine

**NyayaSetu** is an AI-powered legal research tool built to detect conflicts between Supreme Court of India judgments. It uses an advanced multi-stage Retrieval-Augmented Generation (RAG) pipeline to analyze landmark and citing judgments, classify their relationships (AFFIRMS / NARROWS / OVERRULES), and present a beautifully structured legal analysis.

> **Current scope:** Built to analyze Supreme Court judgments, currently focused on employment & labor law.

---

## ✨ Features

- **Knowledge Base Ingestion Pipeline** — A dedicated drag-and-drop UI to ingest new PDFs (Landmark or Citing judgments) dynamically into the vector store.
- **Conflict Detection Engine** — Explicitly calls out when a new judgment overrules or narrows an older one, complete with a Legal Evolution Timeline.
- **Blackish Premium UI** — A modern, single-file React SPA with a stunning dark theme, glassmorphism elements, and smooth micro-animations.
- **Two-Stage Retrieval** — Combines BM25 keyword search with dense vector search, passing results through a Cross-Encoder re-ranker for maximum accuracy.

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────┐
│                   React Frontend                     │
│          (Single-file SPA · Tailwind CSS)            │
│         [ Assistant ]   |   [ Knowledge Base ]       │
├──────────────────────────────────────────────────────┤
│                   FastAPI Backend                    │
│   /query · /ingest_upload · /documents · /health     │
├──────────────────────────────────────────────────────┤
│                  NyayaSetu Pipeline                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Document   │  │  NyayaSetu   │  │  Conflict   │  │
│  │  Ingester   │  │  Retriever   │  │  Detector   │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  │
├──────────────────────────────────────────────────────┤
│  ChromaDB (Vector Store) · HuggingFace Embeddings    │
│  BM25 (Keyword Search) · Cross-Encoder Re-ranker     │
│  Groq LLaMA 3.3 70B (LLM)                            │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
cd FinRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the `FinRAG/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start the API server

```bash
python3 api.py
```

The server starts at **http://localhost:8000**. The frontend is served automatically at the root.

---

## 📡 API Endpoints

### `GET /health`
Liveness check.

### `POST /query`
```json
// Request
{ "query": "Can contract workers claim permanent employment?" }

// Response
{
  "answer": "...",
  "conflict_type": "NARROWS",
  "landmark_case": "State Of Haryana vs Piara Singh",
  "citing_case": "Secretary, State Of Karnataka vs Umadevi",
  "conflict_detected": true
}
```

### `POST /ingest_upload`
Multipart Form-Data endpoint for uploading PDFs.
- `file`: The PDF file
- `doc_type`: "landmark" | "citing"
- `doc_id`: Unique identifier (e.g., `Bangalore_water_supply_1978`)
- `parent_id`: The ID of the landmark it cites (if `doc_type` is citing).

### `GET /documents`
Returns a list of all uniquely ingested documents in the vector store with their metadata.

---

## 🖥️ Frontend Overview

The UI is a sleek, blackish premium dashboard located in `frontend/index.html`. It has two main tabs:

### 1. Assistant Tab
- **Chat interface** for querying the RAG pipeline.
- **Conflict badges** — color-coded (🔴 OVERRULES · 🟠 NARROWS · 🟢 AFFIRMS).
- **Legal Evolution Timeline** — horizontal visual showing how the law evolved between judgments.

### 2. Knowledge Base Tab
- **Drag-and-Drop Ingestion:** Upload new PDFs directly from the browser.
- **Classification Dropdown:** Tag documents as Landmark or Citing.
- **Ingested Documents List:** A live view of all judgments currently sitting in ChromaDB.

---

## 📁 Project Structure

```text
FinRAG/
├── main.py              # NyayaSetu RAG pipeline (core logic)
├── api.py               # FastAPI backend
├── .env                 # Environment variables
├── frontend/
│   └── index.html       # React + Tailwind SPA (Blackish Premium)
├── nyayasetu_db/        # ChromaDB persistent storage
├── uploads/             # Temporarily stages uploaded PDFs
└── README.md
```

---

## ⚠️ Disclaimer

NyayaSetu is a research tool and **not a substitute for professional legal advice**. Always consult a qualified legal professional for legal matters.
