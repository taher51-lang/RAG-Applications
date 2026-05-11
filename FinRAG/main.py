import os
import re
from datetime import datetime
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()
import pdfplumber
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
# LangChain Imports
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Models
import chromadb
from sentence_transformers import CrossEncoder

# =====================================================================
# Step 1: Document Ingester
# Class responsibility: Handle the reading, metadata extraction, and
# vector database ingestion for PDFs. Keeps IO and parsing logic isolated.
# =====================================================================
class DocumentIngester:
    def __init__(self, landmark_retriever, citing_retriever):
        # We inject the ParentDocumentRetrievers so the ingester knows
        # where to push the parsed chunks without worrying about ChromaDB internals.
        self.landmark_retriever = landmark_retriever
        self.citing_retriever = citing_retriever

    def extract_text(self, pdf_path):
        """Extracts full raw text from a given PDF file."""
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    
    def extract_metadata(self, text):
        """Extracts basic metadata like case name and date from the top of the judgment."""
        lines = text.strip().split('\n')
        case_name = lines[0].strip() if lines else "unknown"
        
        date_match = re.search(
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|'
            r'August|September|October|November|December)\s+(\d{4})', 
            text
        )
        date = "unknown"
        if date_match:
            date_str = f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}"
            date = datetime.strptime(date_str, "%d %B %Y").strftime("%Y-%m")
            
        return case_name, date

    def ingest_judgment(self, pdf_path, doc_type, doc_id, parent_id=None):
        """Main method to parse PDF, add metadata, and ingest into the retriever."""
        full_text = self.extract_text(pdf_path)
        case_name, date = self.extract_metadata(full_text)
        
        # Self-reference for landmarks
        if parent_id is None:
            parent_id = doc_id
        
        document = Document(
            page_content=full_text,
            metadata={
                "doc_id": doc_id,
                "doc_type": doc_type,
                "parent_id": parent_id,
                "case_name": case_name,
                "court": "Supreme Court of India",
                "date": date,
                "version": "1.0",
                "is_superseded": False,
                "conflict_detected": False
            }
        )
        
        # Route the document to the corresponding retriever
        if doc_type == "landmark":
            self.landmark_retriever.add_documents([document])
            print(f"Ingested landmark: {case_name} | {doc_id} | {date}")
        else:
            self.citing_retriever.add_documents([document])
            print(f"Ingested citing judgment: {case_name} | {doc_id} | {date}")
        
        return doc_id


# =====================================================================
# Step 2: NyayaSetu Retriever
# Class responsibility: Execute retrieval logic, hybrid searches, and 
# mapping queries from citing judgments to landmark queries.
# =====================================================================
class NyayaSetuRetriever:
    def __init__(self, landmark_vectorstore, citing_vectorstore, cross_encoder, llm):
        # We store the vectorstores to fetch documents dynamically for BM25
        self.landmark_vectorstore = landmark_vectorstore
        self.citing_vectorstore = citing_vectorstore
        self.cross_encoder = cross_encoder
        self.llm = llm
        
        # relationship_registry dictates which landmark applies to which citing judgment
        self.relationship_registry = {
            "Secretary_State_Of_Karnataka_And_vs_Umadevi": [
                "Piarasingh_Vs_State_Of_Punjab",
                "Banglore_water_supply_1978"
            ]
        }
        
    def _get_hybrid_retriever(self, vectorstore):
        """Helper to build a hybrid retriever dynamically combining BM25 and Vector search."""
        all_data = vectorstore.get()
        if not all_data['documents']:
            return vectorstore.as_retriever(search_kwargs={"k": 20})
            
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data['documents'], all_data['metadatas'])
        ]
        bm25_retriever = BM25Retriever.from_documents(docs, k=20)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )

    def two_stage_retrieval(self, query, hybrid_retriever, k_final=5, return_scores=False):
        """Executes Stage 1 (Hybrid retrieval) and Stage 2 (Cross-Encoder Re-ranking)."""
        initial_results = hybrid_retriever.invoke(query)
        
        if not initial_results:
            return ([], []) if return_scores else []
        
        pairs = [[query, doc.page_content] for doc in initial_results]
        scores = self.cross_encoder.predict(pairs)
    
        scored_docs = sorted(
            zip(scores, initial_results),
            key=lambda x: x[0],
            reverse=True
        )
    
        top_docs = [doc for score, doc in scored_docs[:k_final]]
        top_scores = [float(score) for score, doc in scored_docs[:k_final]]
    
        if return_scores:
            return top_docs, top_scores
        return top_docs

    def generate_landmark_query(self, user_query, citing_chunk):
        """Uses the LLM to rewrite the query contextually for landmark retrieval."""
        prompt = f"""
        The user asked: {user_query}

        A citing judgment says: {citing_chunk[:300]}

        What legal principle from a 1978 landmark judgment about 
        the definition of 'industry' and employer-employee relationships 
        would be most relevant to this question?

        Write one paragraph as if you are summarizing the relevant 
        section of that landmark judgment.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def nyayasetu_query(self, user_query):
        citing_hybrid = self._get_hybrid_retriever(self.citing_vectorstore)
        citing_results, citing_scores = self.two_stage_retrieval(
            user_query, citing_hybrid, k_final=5, return_scores=True
        )
        
        RELEVANCE_THRESHOLD = 0.3
    
        # fallback 1 — citing not relevant enough
        if not citing_results or citing_scores[0] < RELEVANCE_THRESHOLD:
            landmark_hybrid = self._get_hybrid_retriever(self.landmark_vectorstore)
            landmark_results = self.two_stage_retrieval(user_query, landmark_hybrid, k_final=5)
            return landmark_results, None
        
        top_citing = citing_results[0]
        citing_doc_id = top_citing.metadata['parent_id']
        landmark_doc_ids = self.relationship_registry.get(citing_doc_id)
    
        # fallback 2 — no registry mapping found
        if not landmark_doc_ids:
            landmark_hybrid = self._get_hybrid_retriever(self.landmark_vectorstore)
            landmark_results = self.two_stage_retrieval(user_query, landmark_hybrid, k_final=5)
            return landmark_results, None
        
        landmark_query = self.generate_landmark_query(user_query, top_citing.page_content)
        
        landmark_results = self.landmark_vectorstore.similarity_search(
            landmark_query,
            k=5,
            filter={"parent_id": landmark_doc_ids[0]}

        )
        
        print("DEBUG landmark_results:", landmark_results)
        print("DEBUG citing_results:", citing_results)
        
        return landmark_results, citing_results


# =====================================================================
# Step 3: Conflict Detector
# Class responsibility: Use LLMs to read the retrieved chunks and 
# explicitly detect if judgments conflict, overrule, or affirm each other.
# =====================================================================
class ConflictDetector:
    def __init__(self, llm):
        # Injected LLM dependency
        self.llm = llm

    def detect_and_answer(self, query, landmark_chunks, citing_chunks):
        """Constructs prompt for LLM to identify any conflict between the landmark and citing judgments."""
        landmark_text = "\n".join([c.page_content for c in landmark_chunks]) if landmark_chunks else "None"
        citing_text = "\n".join([c.page_content for c in citing_chunks]) if citing_chunks else "None"
        
        prompt = f"""
You are NyayaSetu, an authoritative Indian legal research assistant. Provide a professional, direct legal answer. Do NOT use phrases like "based on the provided text", "the citing judgment is not found", or "I cannot cite". Speak directly to the user about the law.

AVAILABLE LEGAL CONTEXT:

[LANDMARK JUDGMENT]
{landmark_text}

[CITING JUDGMENT]
{citing_text}

Instructions:
1. Answer the user's question directly using ONLY the provided legal context.
2. If BOTH a Landmark and Citing judgment are present (not "None"), you MUST classify their relationship as exactly one of: AFFIRMS / NARROWS / OVERRULES. Format this strictly as "**Relationship:** [TYPE]".
3. If only one judgment is present, do not output a relationship classification. Just explain the legal position established by the judgment.
4. State the CURRENT legal position clearly.
5. Cite the judgments that are actually provided. Do NOT apologize or complain if a judgment is missing.
6. If the context is empty or irrelevant, simply state: "The current legal database does not contain information to answer this question." Do not guess.

User question: {query}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


# =====================================================================
# Step 4: NyayaSetu (Main Orchestrator)
# Class responsibility: The single entry point for the system. It builds
# all dependencies (embeddings, LLM, Vectorstores) and ties them together.
# =====================================================================
class NyayaSetu:
    def __init__(self):
        # 1. Initialize Embeddings & ChromaDB inside Orchestrator
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./nyayasetu_db")
  

# replace these two lines in __init__:

# with these:
        self.landmark_docstore = create_kv_docstore(
            LocalFileStore("./nyayasetu_db/landmark_docstore")
        )
        self.citing_docstore = create_kv_docstore(
            LocalFileStore("./nyayasetu_db/citing_docstore")
        )
        self.landmark_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="landmark_judgments",
            embedding_function=self.embeddings
        )
        self.citing_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="citing_judgments",
            embedding_function=self.embeddings
        )
        
        # 2. Splitters for chunking
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
        )
        
        # 3. Initialize ParentDocumentRetrievers
        self.landmark_retriever = ParentDocumentRetriever(
            vectorstore=self.landmark_vectorstore,
            docstore=self.landmark_docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        self.citing_retriever = ParentDocumentRetriever(
            vectorstore=self.citing_vectorstore,
            docstore=self.citing_docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        # 4. Initialize ML Models (Cross Encoder & Generative LLM)
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')

        self.llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0
)
        # 5. Initialize the Sub-Components using Dependency Injection
        self.ingester = DocumentIngester(self.landmark_retriever, self.citing_retriever)
        self.retriever = NyayaSetuRetriever(
            self.landmark_vectorstore, 
            self.citing_vectorstore,
            self.cross_encoder,
            self.llm
        )
        self.conflict_detector = ConflictDetector(self.llm)

    def ingest(self, pdf_path, doc_type, doc_id, parent_id=None):
        """Single entry point for ingestion - delegates to DocumentIngester."""
        return self.ingester.ingest_judgment(pdf_path, doc_type, doc_id, parent_id)

    def query(self, user_query):
    # Step 1: Execute retrieval logic
        landmark_results, citing_results = self.retriever.nyayasetu_query(user_query)
    
        l_chunks = landmark_results if landmark_results else []
        c_chunks = citing_results if citing_results else []
    
    # Step 2: Detect conflicts and generate natural language answer
        raw_answer = self.conflict_detector.detect_and_answer(user_query, l_chunks, c_chunks)
    
    # Step 3: Collect contexts for RAGAS
        contexts = [c.page_content for c in l_chunks + c_chunks]
    
    # Step 4: Parse output to extract structured response
        landmark_case = l_chunks[0].metadata.get("case_name") if l_chunks else None
        citing_case = c_chunks[0].metadata.get("case_name") if c_chunks else None
    
        conflict_type = None
        conflict_detected = False
    
    # Fixed conflict detection — regex instead of simple string search
        import re
        match = re.search(
        r'\*{0,2}(?:Relationship|Classification)\*{0,2}[:\s]+([A-Z]+)',
        raw_answer
    )
        if match:
            conflict_type = match.group(1)
            conflict_detected = conflict_type in ["OVERRULES", "NARROWS"]
    
        return {
        "answer": raw_answer,
        "conflict_type": conflict_type,
        "landmark_case": landmark_case,
        "citing_case": citing_case,
        "conflict_detected": conflict_detected,
        "contexts": contexts
    }

    def debug(self, query):
    # check landmark collection
        landmark_data = self.landmark_vectorstore.get()
        landmark_parent_ids = set(m['parent_id'] for m in landmark_data['metadatas'])
        print("LANDMARK parent_ids:", landmark_parent_ids)
    
    # check citing collection
        citing_data = self.citing_vectorstore.get()
        citing_parent_ids = set(m['parent_id'] for m in citing_data['metadatas'])
        print("CITING parent_ids:", citing_parent_ids)
    
    # check retrieval
        citing_hybrid = self.retriever._get_hybrid_retriever(self.citing_vectorstore)
        citing_results, scores = self.retriever.two_stage_retrieval(
        query, citing_hybrid, k_final=5, return_scores=True
        )
    
        if citing_results:
            top = citing_results[0]
            print("Top citing chunk parent_id:", top.metadata['parent_id'])
            print("Top citing score:", scores[0])
            registry_result = self.retriever.relationship_registry.get(top.metadata['parent_id'])
            print("Registry lookup result:", registry_result)
        else:
            print("No citing results found")

    # Example usage
if __name__ == "__main__":
    nyayasetu = NyayaSetu()
    # nyayasetu.ingest("Bangalore_water_supply_1978.pdf", "landmark", "Banglore_water_supply_1978")
    # nyayasetu.ingest("piyara_Singh.pdf", "landmark", "Piarasingh_Vs_State_Of_Punjab")
    # nyayasetu.ingest("Secretary_State_Of_Karnataka_And_vs_Umadevi.PDF", "citing", "Secretary_State_Of_Karnataka_And_vs_Umadevi")
    # response = nyayasetu.query("can contract workers claim permanent employment")
    # print(response)
    # nyayasetu.debug("can contract workers claim permanent employment")
    # nyayasetu.debug("can contract workers claim permanent employment")
    eval_questions = [
    {
        "question": "Can contract workers claim permanent employment?",
        "ground_truth": "Contract workers do not have a fundamental right to claim permanent employment as per Umadevi 2006 which narrowed the position established in Piara Singh 1992."
    },
    {
        "question": "What is the definition of industry under Industrial Disputes Act?",
        "ground_truth": "Industry means any business, trade, undertaking where capital and labour cooperate for production of goods or services as established in Bangalore Water Supply case 1978."
    },
    {
        "question": "Do temporary employees have regularization rights?",
        "ground_truth": "Temporary employees do not have automatic regularization rights. Umadevi 2006 held that initial appointment must follow proper selection process."
    }
    ]

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision
    from datasets import Dataset

# RAGAS Embeddings — reuse your existing embeddings

