---
title: NyayaSetu
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# ⚖️ FinRAG (NyayaSetu) — Indian Legal Research Engine

**FinRAG (NyayaSetu)** is an AI-powered legal research tool built to detect conflicts between Supreme Court of India judgments. It uses an advanced multi-stage Retrieval-Augmented Generation (RAG) pipeline to analyze landmark and citing judgments, classify their relationships (AFFIRMS / NARROWS / OVERRULES), and present a beautifully structured legal analysis.

> **Current scope:** Built to analyze Supreme Court judgments, currently focused on employment & labor law.

---

## ✨ Features

- **Knowledge Base Ingestion Pipeline** — A dedicated drag-and-drop UI to ingest new PDFs (Landmark or Citing judgments) dynamically into the vector store.
- **Conflict Detection Engine** — Explicitly calls out when a new judgment overrules or narrows an older one, complete with a Legal Evolution Timeline.
- **Blackish Premium UI** — A modern, single-file React SPA with a stunning dark theme, glassmorphism elements, and smooth micro-animations.
- **Two-Stage Hybrid Retrieval** — Combines BM25 keyword (sparse) search with Cohere dense vector search using Pinecone, passing results through a Cross-Encoder re-ranker for maximum accuracy.

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
│  Pinecone (Vector Store) · Cohere Embeddings         │
│  Upstash Redis (Docstore)                            │
│  BM25 (Keyword Search) · Cross-Encoder Re-ranker     │
│  Groq LLaMA 3.3 70B (LLM)                            │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
You will need API keys/credentials for the following services:
- **Groq**: For the LLaMA-3.3-70B model.
- **Cohere**: For embedding models (`embed-english-v3.0`).
- **Pinecone**: For the serverless vector database (create an index named `nyaya-setu` with 1024 dimensions for Cohere embeddings).
- **Upstash Redis**: For document storage.

### 2. Clone & Install Dependencies

```bash
cd FinRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root `FinRAG/` directory and populate it with your keys:

```env
# LLM Provider
GROQ_API_KEY=your_groq_api_key_here

# Embeddings
COHERE_API_KEY=your_cohere_api_key_here

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key_here

# Document Store (Upstash Redis)
UPSTASH_REDIS_REST_URL=your_upstash_redis_url_here
UPSTASH_REDIS_REST_TOKEN=your_upstash_redis_token_here
```

### 4. Start the API Server

Use Uvicorn to run the FastAPI backend server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at **http://localhost:8000**. The frontend is served automatically at the root (`/`). 
Open your browser and navigate to `http://localhost:8000` to interact with the assistant.

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
  "conflict_detected": true,
  "contexts": [...]
}
```

### `POST /ingest_upload`
Multipart Form-Data endpoint for uploading PDFs.
- `file`: The PDF file
- `doc_type`: "landmark" | "citing"
- `doc_id`: Unique identifier (e.g., `Bangalore_water_supply_1978`)
- `parent_id`: The ID of the landmark it cites (if `doc_type` is citing).

### `GET /documents`
Returns a list of all uniquely ingested documents in the vector store and Redis docstore with their metadata.

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
- **Ingested Documents List:** A live view of all judgments currently sitting in Redis/Pinecone.

---

## 📁 Project Structure

```text
FinRAG/
├── main.py              # NyayaSetu RAG pipeline (core logic)
├── api.py               # FastAPI backend
├── .env                 # Environment variables
├── frontend/
│   └── index.html       # React + Tailwind SPA (Blackish Premium)
├── uploads/             # Temporarily stages uploaded PDFs
└── README.md
```

---

## ⚠️ Disclaimer

FinRAG (NyayaSetu) is a research tool and **not a substitute for professional legal advice**. Always consult a qualified legal professional for legal matters.
