"""
NyayaSetu FastAPI Backend
=========================
Exposes the NyayaSetu RAG pipeline as a REST API.

Endpoints:
  POST /query   — structured legal query
  POST /ingest  — ingest a new PDF judgment
  GET  /health  — liveness check
"""

import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Import NyayaSetu from existing main module ──────────────────────
from main import NyayaSetu

# ── Pydantic request/response schemas ──────────────────────────────
class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    pdf_path: str
    doc_type: str
    doc_id: str
    parent_id: str | None = None

# ── Global singleton ────────────────────────────────────────────────
nyayasetu_instance: NyayaSetu | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the NyayaSetu pipeline once at startup."""
    global nyayasetu_instance
    print("🔄 Initializing NyayaSetu pipeline …")
    nyayasetu_instance = NyayaSetu()
    print("✅ NyayaSetu ready.")
    yield
    print("🛑 Shutting down NyayaSetu.")
    nyayasetu_instance = None

# ── FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title="NyayaSetu API",
    description="Indian Legal Research Assistant — conflict detection between Supreme Court judgments",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow all origins ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest):
    if not nyayasetu_instance:
        raise HTTPException(status_code=503, detail="NyayaSetu pipeline is not initialized yet.")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        result = nyayasetu_instance.query(request.query)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/ingest")
async def ingest(request: IngestRequest):
    if not nyayasetu_instance:
        raise HTTPException(status_code=503, detail="NyayaSetu pipeline is not initialized yet.")

    if not os.path.isfile(request.pdf_path):
        raise HTTPException(status_code=400, detail=f"File not found: {request.pdf_path}")

    try:
        doc_id = nyayasetu_instance.ingest(
            pdf_path=request.pdf_path,
            doc_type=request.doc_type,
            doc_id=request.doc_id,
            parent_id=request.parent_id,
        )
        return {"doc_id": doc_id, "status": "ingested"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ingest_upload")
async def ingest_upload(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    doc_id: str = Form(...),
    parent_id: str = Form(None)
):
    if not nyayasetu_instance:
        raise HTTPException(status_code=503, detail="NyayaSetu pipeline is not initialized yet.")
    
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(file.file, buffer)
        
    try:
        res_doc_id = nyayasetu_instance.ingest(
            pdf_path=file_path,
            doc_type=doc_type,
            doc_id=doc_id,
            parent_id=parent_id if parent_id else None,
        )
        return {"doc_id": res_doc_id, "status": "ingested", "filename": file.filename}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/documents")
async def get_documents():
    if not nyayasetu_instance:
        raise HTTPException(status_code=503, detail="NyayaSetu pipeline is not initialized yet.")
        
    documents = []
    
    # Get Landmark documents
    try:
        landmark_data = nyayasetu_instance.landmark_vectorstore.get()
        if landmark_data and 'metadatas' in landmark_data:
            for meta in landmark_data['metadatas']:
                if meta and meta.get('doc_id'):
                    doc = {
                        "doc_id": meta.get('doc_id'),
                        "doc_type": "landmark",
                        "case_name": meta.get('case_name', 'Unknown'),
                        "date": meta.get('date', 'Unknown')
                    }
                    if doc not in documents:
                        documents.append(doc)
    except Exception as e:
        print(f"Error fetching landmark docs: {e}")
        
    # Get Citing documents
    try:
        citing_data = nyayasetu_instance.citing_vectorstore.get()
        if citing_data and 'metadatas' in citing_data:
            for meta in citing_data['metadatas']:
                if meta and meta.get('doc_id'):
                    doc = {
                        "doc_id": meta.get('doc_id'),
                        "parent_id": meta.get('parent_id'),
                        "doc_type": "citing",
                        "case_name": meta.get('case_name', 'Unknown'),
                        "date": meta.get('date', 'Unknown')
                    }
                    if doc not in documents:
                        documents.append(doc)
    except Exception as e:
        print(f"Error fetching citing docs: {e}")
        
    # Filter duplicates by doc_id
    unique_docs = {doc['doc_id']: doc for doc in documents}.values()
    return {"documents": list(unique_docs)}

# ── Serve React frontend ───────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
