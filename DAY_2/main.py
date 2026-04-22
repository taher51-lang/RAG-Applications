# ============================================================
# FastAPI RAG Backend
# If you know Flask, think of FastAPI as Flask but:
# - Faster
# - Built in data validation
# - Auto generates API docs at http://localhost:8000/docs
# ============================================================

# --- IMPORTS ---

# FastAPI is like Flask's app
# UploadFile is for handling file uploads (like request.files in Flask)
# File is a helper that tells FastAPI to expect a file
# BackgroundTasks lets us run slow tasks without blocking the response
from fastapi import FastAPI, UploadFile, File, BackgroundTasks

# CORSMiddleware — same concept as Flask-CORS
# Allows your HTML frontend to talk to this backend
from fastapi.middleware.cors import CORSMiddleware

# BaseModel is like a schema/validator for request body
# In Flask you do request.json.get('query') manually
# In FastAPI you define a class and it validates automatically
from pydantic import BaseModel

# Standard libraries
import uuid        # generates unique session IDs
import tempfile    # creates temporary files
import os          # file operations

# LangChain imports — same as before
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.retrievers import 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)
# ============================================================
# APP SETUP
# ============================================================

# Flask:   app = Flask(__name__)
# FastAPI: app = FastAPI()
# That simple.
app = FastAPI()

# CORS setup — allows frontend HTML file to call this API
# Flask: from flask_cors import CORS; CORS(app)
# FastAPI: add middleware like this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins (fine for development)
    allow_methods=["*"],       # allow GET, POST, DELETE etc
    allow_headers=["*"],       # allow all headers
)


# ============================================================
# SHARED OBJECTS
# (created once when server starts, reused for every request)
# ============================================================

# Embedding model — loaded once, used for every upload and query
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-base-en-v1.5",
#     model_kwargs={'device': 'mps'}  # Uses your M1's GPU/Neural Engine
# )

# LLM — loaded once
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
# Prompt template
prompt = PromptTemplate(
    template="""You are a helpful assistant. 
    Answer the question using only the context provided.
    If the answer is not in the context, say 'I could not find this in the document.'
    
    Context: {context}
    
    Question: {question}
    
    """,
    input_variables=["context", "question"]
)

# Chain — same as before

# Sessions dictionary — stores info about each user's upload
# In production you'd use Redis or a database instead
# Think of it like Flask's session but stored in memory
# Format: { "session_id": { "status": "ready", "collection": "session_id" } }
sessions = {}


# ============================================================
# REQUEST MODELS
# ============================================================

# In Flask you do: data = request.get_json(); query = data['query']
# In FastAPI you define a class — it validates automatically
# If 'query' or 'session_id' is missing, FastAPI returns a 422 error automatically
class ChatRequest(BaseModel):
    query: str
    session_id: str


# ============================================================
# BACKGROUND TASK — PDF PROCESSING
# ============================================================

# This function runs in the background after we return the response
# So the user gets session_id immediately and doesn't wait 5 minutes
def process_pdf(tmp_path: str, session_id: str):
    """
    Loads PDF, chunks it, embeds it into ChromaDB.
    Runs in background so user doesn't wait.
    """
    try:
        # Update status — frontend can poll this
        sessions[session_id]["status"] = "processing"

        # Load PDF lazily — one page at a time (memory efficient for large PDFs)
        loader = PyPDFLoader(tmp_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = []
        total_pages = 0

        for doc in loader.lazy_load():
            # Skip pages with very little text (image-only pages)
            if len(doc.page_content.strip()) < 50:
                print("HII")
                continue

            # Add session_id to metadata so we can filter by user later
            doc.metadata["session_id"] = session_id

            # Split this page into chunks
            page_chunks = splitter.split_documents([doc])
            chunks.extend(page_chunks)
            total_pages += 1
            if(len(chunks)>30):
                 
                 vectorStore = Chroma(
                    collection_name=session_id,   # unique per user
                    persist_directory="chromadb",
                    embedding_function=embeddings
                 )
                 vectorStore.add_documents(chunks)

                 chunks=[]

            print(total_pages)
            # Update progress in sessions dict
            sessions[session_id]["pages_processed"] = total_pages

        # Store chunks in ChromaDB
        # Each user gets their OWN collection named after their session_id
        # This way users never see each other's documents
       

        # Update status to ready
        sessions[session_id]["status"] = "ready"
        sessions[session_id]["total_chunks"] = len(chunks)
        sessions[session_id]["total_pages"] = total_pages

        print(f"Session {session_id[:8]}... ready — {len(chunks)} chunks from {total_pages} pages")

    except Exception as e:
        # If something goes wrong, update status to failed
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)
        print(f"Error processing PDF: {e}")

    finally:
        # Always delete the temporary PDF file
        # We never store user PDFs permanently
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# ROUTES
# ============================================================

# --- HEALTH CHECK ---
# Flask:   @app.route('/')
# FastAPI: @app.get('/')
# The HTTP method is part of the decorator — cleaner than Flask
@app.get("/")
def home():
    # In Flask: return jsonify({"message": "..."})
    # In FastAPI: just return a dict — it auto converts to JSON
    return {"message": "RAG API is running"}


# --- UPLOAD ROUTE ---
# Flask:   @app.route('/upload', methods=['POST'])
# FastAPI: @app.post('/upload')
@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,  # FastAPI injects this automatically
    file: UploadFile = File(...)        # File(...) means required file upload
):
    """
    Accepts a PDF, starts processing in background, returns session_id immediately.
    Frontend uses session_id for all future chat requests.
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        # In Flask: return jsonify({"error": "..."}), 400
        # In FastAPI: just return dict — or use HTTPException for proper error codes
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Generate unique session ID for this user's upload
    session_id = str(uuid.uuid4())

    # Save PDF to a temporary file
    # We read it, save it, then delete it after processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()   # 'await' because FastAPI is async
        tmp.write(content)
        tmp_path = tmp.name           # something like /tmp/abc123.pdf

    # Register session with initial status
    sessions[session_id] = {
        "status": "processing",
        "filename": file.filename,
        "pages_processed": 0
    }

    # Add PDF processing to background tasks
    # This returns IMMEDIATELY — processing happens in background
    # User gets session_id right away instead of waiting 5 minutes
    background_tasks.add_task(process_pdf, tmp_path, session_id)

    return {
        "session_id": session_id,
        "message": "Upload received. Processing started.",
        "status": "processing"
    }


# --- STATUS CHECK ROUTE ---
# Frontend polls this every 2 seconds to check if PDF is ready
@app.get("/status/{session_id}")
def status(session_id: str):
    """
    Returns current processing status for a session.
    Frontend polls this until status is 'ready'.
    """
    # {session_id} in the route is a path parameter
    # Flask:   @app.route('/status/<session_id>')  def status(session_id):
    # FastAPI: @app.get('/status/{session_id}')    def status(session_id: str):
    # FastAPI adds type hint — auto validates it's a string

    if session_id not in sessions:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")

    return sessions[session_id]


# --- CHAT ROUTE ---
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Accepts query + session_id, retrieves relevant chunks, returns LLM answer.
    """
    # FastAPI automatically validated req.query and req.session_id exist
    # If either is missing the request fails before reaching this function

    # Check session exists
    if req.session_id not in sessions:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found. Please upload your PDF again.")

    # Check PDF is ready
    session = sessions[req.session_id]
    if session["status"] == "processing":
        return {"response": "Your document is still being processed. Please wait a moment."}

    if session["status"] == "failed":
        return {"response": f"Document processing failed: {session.get('error', 'Unknown error')}"}

    # Load the vector store for this specific user
    vectorStore = Chroma(
        collection_name=req.session_id,
        persist_directory="chromadb",
        embedding_function=embeddings
    )

    # Retrieve relevant chunks
    print(f"query {req.query}")
    retriver = vectorStore.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.5}
)

    # Combine chunks into context
    chain = {"context": lambda x: retriver.invoke(x["question"]),"question":RunnablePassthrough()} |prompt | chat_model | StrOutputParser()

    # Run through LLM
    output = chain.invoke({
        "question": req.query
    })

    return {"response": output}


# --- DELETE SESSION ---
# Clean up when user is done — delete their embeddings
@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Deletes a user's ChromaDB collection and session data.
    Call this when user closes the chat or after a timeout.
    """
    if session_id not in sessions:
        return {"message": "Session not found"}

    try:
        # Delete ChromaDB collection for this session
        import chromadb
        client = chromadb.PersistentClient(path="chromadb")
        client.delete_collection(session_id)
    except Exception as e:
        print(f"Error deleting collection: {e}")

    # Remove from sessions dict
    del sessions[session_id]

    return {"message": "Session deleted successfully"}


# ============================================================
# RUN THE APP
# ============================================================

# Flask:   if __name__ == '__main__': app.run(debug=True)
# FastAPI: run with uvicorn from terminal
#          uvicorn main:app --reload
#
# After running, visit:
# http://localhost:8000/docs  → Auto generated API documentation (Swagger UI)
# http://localhost:8000       → Health check