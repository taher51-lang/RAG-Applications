# 📄 DocChat — Chat with your PDFs using Multi-Strategy RAG

> A **Retrieval-Augmented Generation** application that lets you upload any PDF and chat with it using **two different retrieval strategies** — switch between them in real-time from the UI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?logo=chainlink&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?logo=databricks&logoColor=white)

---

## ✨ What Makes This Different?

Most RAG tutorials show you **one** retrieval method. This project implements **two fundamentally different strategies** and lets the user **toggle between them mid-conversation** — so you can actually feel the difference.

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **⚡ MMR** (Maximal Marginal Relevance) | Retrieves semantically similar chunks, then **re-ranks to maximize diversity** — reducing redundancy | Questions where you want **broad coverage** across the document |
| **🔀 Hybrid** (BM25 + Vector Ensemble) | Combines **keyword matching** (BM25/TF-IDF) with **semantic search** (vector similarity), then fuses results with weighted ensemble | Questions with **specific terms** or technical vocabulary |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                     │
│  ┌──────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  Upload   │  │ Strategy Toggle│  │   Chat Window   │  │
│  │  Zone     │  │  MMR / Hybrid  │  │                 │  │
│  └──────────┘  └────────────────┘  └─────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP (REST API)
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                         │
│                                                          │
│  POST /upload ──→ Background PDF Processing              │
│  GET  /status ──→ Poll processing status                 │
│  POST /chat   ──→ Strategy Router (if/else)              │
│                     │                                    │
│            ┌────────┴────────┐                           │
│            ▼                 ▼                            │
│     ┌──────────┐     ┌──────────────┐                    │
│     │   MMR    │     │   Hybrid     │                    │
│     │ Retriever│     │  Ensemble    │                    │
│     └────┬─────┘     │ ┌──────────┐│                    │
│          │           │ │  BM25    ││                    │
│          │           │ │ (keyword)││                    │
│          │           │ └──────────┘│                    │
│          │           │ ┌──────────┐│                    │
│          │           │ │  Vector  ││                    │
│          │           │ │(semantic)││                    │
│          │           │ └──────────┘│                    │
│          │           └──────┬──────┘                    │
│          │                  │                            │
│          └────────┬─────────┘                            │
│                   ▼                                      │
│         ┌──────────────────┐                             │
│         │ LangChain Chain  │                             │
│         │ Context + Prompt │──→ Gemini 2.5 Flash ──→ 💬 │
│         └──────────────────┘                             │
│                                                          │
│  Storage: ChromaDB (vectors) + In-memory (BM25 chunks)  │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Google AI API key](https://aistudio.google.com/apikey) (for Gemini)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/DocChat-RAG.git
cd DocChat-RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run

```bash
uvicorn main:app --reload
```

Then open **`index.html`** in your browser. That's it!

- **API docs** → [http://localhost:8000/docs](http://localhost:8000/docs) (auto-generated Swagger UI)
- **Health check** → [http://localhost:8000](http://localhost:8000)

---

## 🎯 How It Works

### 1. Upload
Drop a PDF → it's chunked (1000 chars, 100 overlap) and embedded into ChromaDB in the background. You get a `session_id` immediately.

### 2. Choose Strategy
Toggle between **MMR** and **Hybrid** in the sidebar — your choice is sent with every chat request.

### 3. Chat
The backend routes your query through the selected retriever:

```python
if req.retrieval_strategy == "hybrid":
    # BM25 keyword retriever + vector retriever → EnsembleRetriever
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.3, 0.7]
    )
else:
    # MMR: relevance with diversity
    retriever = vectorStore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "lambda_mult": 0.5}
    )
```

The retrieved context is then fed to **Gemini 2.5 Flash** via a LangChain chain.

---

## 📁 Project Structure

```
DocChat-RAG/
├── main.py              # FastAPI backend with dual retrieval
├── index.html           # Frontend UI with strategy toggle
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `GOOGLE_API_KEY` | Your Google AI Studio API key for Gemini |
| `HF_TOKEN` | *(Optional)* HuggingFace token for gated models |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload PDF (multipart form) |
| `GET` | `/status/{session_id}` | Poll processing status |
| `POST` | `/chat` | Send query with `retrieval_strategy` |
| `DELETE` | `/session/{session_id}` | Cleanup session data |

### Chat Request Body
```json
{
  "query": "What is self-attention?",
  "session_id": "uuid-here",
  "retrieval_strategy": "mmr"
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI + Uvicorn |
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | HuggingFace `bge-small-en-v1.5` |
| **Vector Store** | ChromaDB |
| **Keyword Search** | BM25 (rank-bm25) |
| **Ensemble** | LangChain EnsembleRetriever |
| **Frontend** | Vanilla HTML/CSS/JS |

---

## 📝 License

MIT

---

**Built with ☕ and curiosity.**
