# ============================================================
# DocChat — RAG Backend with Multi-Strategy Retrieval
#
# Supports two retrieval strategies:
#   1. MMR  (Maximal Marginal Relevance)    — balances relevance + diversity
#   2. Hybrid (BM25 + Vector Ensemble)      — keyword + semantic fusion
#
# Stack: FastAPI · LangChain · ChromaDB · Gemini · HuggingFace
# ============================================================

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import uuid
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Models & Prompt
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template="""You are a helpful assistant. 
    Answer the question using only the context provided.
    If the answer is not in the context, say 'I could not find this in the document.'
    
    Context: {context}
    
    Question: {question}
    
    """,
    input_variables=["context", "question"]
)

# ============================================================
# App Setup
# ============================================================

app = FastAPI(
    title="DocChat RAG API",
    description="Chat with your PDFs using MMR or Hybrid retrieval strategies",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}


class ChatRequest(BaseModel):
    query: str
    session_id: str
    retrieval_strategy: Optional[str] = "mmr"  # "mmr" or "hybrid"


# ============================================================
# PDF Processing (Background Task)
# ============================================================

def process_pdf(tmp_path: str, session_id: str):
    """Loads PDF, chunks it, embeds into ChromaDB, stores raw chunks for BM25."""
    try:
        sessions[session_id]["status"] = "processing"

        loader = PyPDFLoader(tmp_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        all_chunks = []
        batch_chunks = []
        total_pages = 0

        for doc in loader.lazy_load():
            if len(doc.page_content.strip()) < 50:
                continue

            doc.metadata["session_id"] = session_id
            page_chunks = splitter.split_documents([doc])
            all_chunks.extend(page_chunks)
            batch_chunks.extend(page_chunks)
            total_pages += 1

            if len(batch_chunks) > 30:
                vectorStore = Chroma(
                    collection_name=session_id,
                    persist_directory="chromadb",
                    embedding_function=embeddings
                )
                vectorStore.add_documents(batch_chunks)
                batch_chunks = []

            sessions[session_id]["pages_processed"] = total_pages

        if batch_chunks:
            vectorStore = Chroma(
                collection_name=session_id,
                persist_directory="chromadb",
                embedding_function=embeddings
            )
            vectorStore.add_documents(batch_chunks)

        # Store raw chunks for BM25 keyword retrieval
        sessions[session_id]["chunks"] = all_chunks
        sessions[session_id]["status"] = "ready"
        sessions[session_id]["total_chunks"] = len(all_chunks)
        sessions[session_id]["total_pages"] = total_pages

        print(f"✓ Session {session_id[:8]}... ready — {len(all_chunks)} chunks from {total_pages} pages")

    except Exception as e:
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)
        print(f"✗ Error processing PDF: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# Routes
# ============================================================

@app.get("/")
def home():
    return {"message": "DocChat RAG API is running", "strategies": ["mmr", "hybrid"]}


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF for processing. Returns session_id for subsequent requests."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    session_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    sessions[session_id] = {
        "status": "processing",
        "filename": file.filename,
        "pages_processed": 0
    }

    background_tasks.add_task(process_pdf, tmp_path, session_id)

    return {
        "session_id": session_id,
        "message": "Upload received. Processing started.",
        "status": "processing"
    }


@app.get("/status/{session_id}")
def status(session_id: str):
    """Poll processing status. Returns 'processing', 'ready', or 'failed'."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session_data = {k: v for k, v in sessions[session_id].items() if k != "chunks"}
    return session_data


# ============================================================
# Chat — Multi-Strategy Retrieval
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Query your document with a chosen retrieval strategy.
    
    Strategies:
      - mmr:    Maximal Marginal Relevance (diversity + relevance)
      - hybrid: BM25 keyword + vector semantic ensemble
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")

    session = sessions[req.session_id]

    if session["status"] == "processing":
        return {"response": "Document is still being processed. Please wait."}

    if session["status"] == "failed":
        return {"response": f"Processing failed: {session.get('error', 'Unknown error')}"}

    vectorStore = Chroma(
        collection_name=req.session_id,
        persist_directory="chromadb",
        embedding_function=embeddings
    )

    # ── Strategy Selection ─────────────────────────────────────

    if req.retrieval_strategy == "hybrid":
        # Hybrid: BM25 (keyword) + Vector (semantic) ensemble
        vector_retriever = vectorStore.as_retriever(search_kwargs={"k": 4})

        chunks = session.get("chunks", [])
        if not chunks:
            return {"response": "No chunks available for hybrid search. Try re-uploading."}

        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k = 4

        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.3, 0.7]
        )
    else:
        # MMR: Maximal Marginal Relevance
        retriever = vectorStore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "lambda_mult": 0.5}
        )

    # ── Chain ──────────────────────────────────────────────────

    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    output = chain.invoke({"question": req.query})

    return {"response": output, "strategy_used": req.retrieval_strategy}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete session data and ChromaDB collection."""
    if session_id not in sessions:
        return {"message": "Session not found"}

    try:
        import chromadb
        client = chromadb.PersistentClient(path="chromadb")
        client.delete_collection(session_id)
    except Exception as e:
        print(f"Error deleting collection: {e}")

    del sessions[session_id]
    return {"message": "Session deleted successfully"}
