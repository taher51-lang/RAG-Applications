# ============================================================
# FastAPI RAG Backend —
# Supports two retrieval strategies:
#   1. MMR (Maximal Marginal Relevance)  — default
#   2. Hybrid Reranking (BM25 keyword + vector ensemble)
# ============================================================

# --- IMPORTS ---
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import uuid
import tempfile
import os

# LangChain core
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Hybrid retrieval imports (DAY 3)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# SHARED OBJECTS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

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
# APP SETUP
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sessions dictionary — stores info about each user's upload
# Also stores the raw chunks needed for BM25 (keyword retrieval)
sessions = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    query: str
    session_id: str
    retrieval_strategy: Optional[str] = "mmr"  # "mmr" or "hybrid"


# ============================================================
# BACKGROUND TASK — PDF PROCESSING
# ============================================================

def process_pdf(tmp_path: str, session_id: str):
    """
    Loads PDF, chunks it, embeds it into ChromaDB.
    Also stores raw chunks in session for BM25 retrieval.
    """
    try:
        sessions[session_id]["status"] = "processing"

        loader = PyMuPDFLoader(tmp_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        all_chunks = []  # keep ALL chunks for BM25
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

            # Batch insert into ChromaDB every 30 chunks
            if len(batch_chunks) > 30:
                vectorStore = Chroma(
                    collection_name=session_id,
                    persist_directory="chromadb",
                    embedding_function=embeddings
                )
                vectorStore.add_documents(batch_chunks)
                batch_chunks = []

            print(total_pages)
            sessions[session_id]["pages_processed"] = total_pages

        # Insert any remaining chunks
        if batch_chunks:
            vectorStore = Chroma(
                collection_name=session_id,
                persist_directory="chromadb",
                embedding_function=embeddings
            )
            vectorStore.add_documents(batch_chunks)

        # ============================================================
        # IMPORTANT: Store raw chunks in session for BM25 retrieval
        # BM25 needs the original Document objects (keyword matching)
        # ============================================================
        sessions[session_id]["chunks"] = all_chunks

        sessions[session_id]["status"] = "ready"
        sessions[session_id]["total_chunks"] = len(all_chunks)
        sessions[session_id]["total_pages"] = total_pages

        print(f"Session {session_id[:8]}... ready — {len(all_chunks)} chunks from {total_pages} pages")

    except Exception as e:
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)
        print(f"Error processing PDF: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {"message": "RAG API is running — MMR + Hybrid Reranking"}


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts a PDF, starts processing in background, returns session_id.
    """
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
    """Returns current processing status for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return a copy without the chunks (they can be huge)
    session_data = {k: v for k, v in sessions[session_id].items() if k != "chunks"}
    return session_data


# ============================================================
# CHAT ROUTE — The if/else magic happens here!
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Accepts query + session_id + retrieval_strategy.
    Uses MMR or Hybrid Reranking based on user's choice.
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload your PDF again.")

    session = sessions[req.session_id]

    if session["status"] == "processing":
        return {"response": "Your document is still being processed. Please wait a moment."}

    if session["status"] == "failed":
        return {"response": f"Document processing failed: {session.get('error', 'Unknown error')}"}

    # Load the vector store for this user
    vectorStore = Chroma(
        collection_name=req.session_id,
        persist_directory="chromadb",
        embedding_function=embeddings
    )

    print(f"[{req.retrieval_strategy.upper()}] query: {req.query}")

    # ============================================================
    # THE IF/ELSE — Choose retrieval strategy
    # ============================================================

    if req.retrieval_strategy == "hybrid":
        # -------------------------------------------------------
        # HYBRID RERANKING (DAY 3)
        # Combines:
        #   1. Vector retriever (semantic similarity from ChromaDB)
        #   2. BM25 retriever  (keyword matching — TF-IDF style)
        # Then ensembles them with configurable weights
        # -------------------------------------------------------

        # Vector retriever — standard similarity search
        vector_retriever = vectorStore.as_retriever(
            search_kwargs={"k": 4}
        )

        # BM25 keyword retriever — uses stored raw chunks
        chunks = session.get("chunks", [])
        if not chunks:
            return {"response": "No chunks available for hybrid search. Try re-uploading."}

        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k = 4

        # Ensemble — merge results from both retrievers
        # weights: 30% vector (semantic) + 70% keyword (BM25)
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.3, 0.7]
        )

        print("  → Using Hybrid: BM25(0.7) + Vector(0.3) ensemble")

    else:
        # -------------------------------------------------------
        # MMR — Maximal Marginal Relevance (DAY 2)
        # Balances relevance with diversity
        # lambda_mult: 0 = max diversity, 1 = max relevance
        # -------------------------------------------------------

        retriever = vectorStore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "lambda_mult": 0.5}
        )

        print("  → Using MMR: k=6, lambda=0.5")

    # ============================================================
    # Build and run the chain (same for both strategies)
    # ============================================================

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

    return {
        "response": output,
        "strategy_used": req.retrieval_strategy
    }


# --- DELETE SESSION ---
@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Deletes a user's ChromaDB collection and session data."""
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


# ============================================================
# RUN: uvicorn main:app --reload
# ============================================================
