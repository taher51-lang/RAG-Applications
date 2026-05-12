import os
import sys
print(f"Python: {sys.version}", flush=True)
print("Starting api.py import...", flush=True)
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.storage import UpstashRedisByteStore
from upstash_redis import Redis
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()
redis = Redis.from_env()
import pdfplumber
from langchain_classic.storage import LocalFileStore
from langchain_pinecone import PineconeVectorStore
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

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever

# =====================================================================
# Step 1: Document Ingester
# Class responsibility: Handle the reading, metadata extraction, and
# vector database ingestion for PDFs. Keeps IO and parsing logic isolated.
# =====================================================================
class DocumentIngester:
    def __init__(self, landmark_hybrid, citing_hybrid, landmark_docstore, citing_docstore, child_splitter):
        self.landmark_hybrid = landmark_hybrid
        self.citing_hybrid = citing_hybrid
        self.landmark_docstore = landmark_docstore
        self.citing_docstore = citing_docstore
        self.child_splitter = child_splitter

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
        
        parent_doc = Document(
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
        
        # Save parent to docstore
        if doc_type == "landmark":
            # RIGHT: Passing the encoded content as bytes
            self.landmark_docstore.mset([(doc_id, parent_doc.page_content.encode("utf-8"))])
            hybrid_retriever = self.landmark_hybrid
            print(f"Ingested landmark parent: {case_name} | {doc_id} | {date}")
        else:
            # The corrected line for your citing judgments
            self.citing_docstore.mset([(doc_id, parent_doc.page_content.encode("utf-8"))])
            hybrid_retriever = self.citing_hybrid
            print(f"Ingested citing parent: {case_name} | {doc_id} | {date}")
            
        # Chunking
        chunks = self.child_splitter.split_documents([parent_doc])
        texts = []
        metadatas = []
        for chunk in chunks:
            chunk.metadata['parent_id'] = doc_id
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            
        # Add chunks to hybrid retriever
        if texts:
            # We determine the correct namespace string based on the doc_type
            target_namespace = "landmark_judgments" if doc_type == "landmark" else "citing_judgments"
            
            # CRITICAL FIX: Pass the namespace explicitly here
            hybrid_retriever.add_texts(
                texts, 
                metadatas=metadatas, 
                namespace=target_namespace
            )
            print(f"✅ Ingested {len(texts)} chunks into namespace: {target_namespace}")
        
        return doc_id


# =====================================================================
# Step 2: NyayaSetu Retriever
# Class responsibility: Execute retrieval logic, hybrid searches, and 
# mapping queries from citing judgments to landmark queries.
# =====================================================================
class NyayaSetuRetriever:
    def __init__(self, landmark_hybrid, citing_hybrid, cross_encoder, llm):
        self.landmark_hybrid = landmark_hybrid
        self.citing_hybrid = citing_hybrid
        self.cross_encoder = cross_encoder
        self.llm = llm
        
        # relationship_registry dictates which landmark applies to which citing judgment
        self.relationship_registry = {
            "Secretary_State_Of_Karnataka_And_vs_Umadevi": [
                "Piarasingh_Vs_State_Of_Punjab",
                "Bangalore_water_supply_1978"
            ]
        }

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
        citing_results, citing_scores = self.two_stage_retrieval(
            user_query, self.citing_hybrid, k_final=5, return_scores=True
        )
        
        RELEVANCE_THRESHOLD = 0.3
    
        # fallback 1 — citing not relevant enough
        if not citing_results or citing_scores[0] < RELEVANCE_THRESHOLD:
            landmark_results = self.two_stage_retrieval(user_query, self.landmark_hybrid, k_final=5)
            return landmark_results, None
        
        top_citing = citing_results[0]
        citing_doc_id = top_citing.metadata['parent_id']
        landmark_doc_ids = self.relationship_registry.get(citing_doc_id)
    
        # fallback 2 — no registry mapping found
        if not landmark_doc_ids:
            landmark_results = self.two_stage_retrieval(user_query, self.landmark_hybrid, k_final=5)
            return landmark_results, None
        
        landmark_query = self.generate_landmark_query(user_query, top_citing.page_content)
        
        # For Pinecone Hybrid Search, we get relevant parent documents
        # and then filter them by parent_id. 
        landmark_results = self.two_stage_retrieval(landmark_query, self.landmark_hybrid, k_final=10)
        filtered_landmark_results = [doc for doc in landmark_results if doc.metadata.get("parent_id") in landmark_doc_ids]
        
        return filtered_landmark_results[:5], citing_results


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
        # 1. Initialize Embeddings & Pinecone Vector Database
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "nyaya-setu"
        index = pc.Index(index_name)
        
        # BM25 Encoder for sparse embeddings
        self.bm25_encoder = BM25Encoder().default()
        
        self.landmark_docstore = UpstashRedisByteStore(
            client=redis, 
            ttl=None, 
            namespace="landmark"
        )

        self.citing_docstore = UpstashRedisByteStore(
            client=redis, 
            ttl=None, 
            namespace="citing"
        )

        # 2. Splitters for chunking
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
        )

        # 3. Initialize PineconeHybridSearchRetrievers
        self.landmark_hybrid_retriever = PineconeHybridSearchRetriever(
            embeddings=self.embeddings,
            sparse_encoder=self.bm25_encoder,
            index=index,
            namespace="landmark_judgments",
            top_k=20
        )

        self.citing_hybrid_retriever = PineconeHybridSearchRetriever(
            embeddings=self.embeddings,
            sparse_encoder=self.bm25_encoder,
            index=index,
            namespace="citing_judgments",
            top_k=20
        )
        
        # 4. Initialize ML Models (Cross Encoder & Generative LLM)
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0
        )
        
        # 5. Initialize the Sub-Components using Dependency Injection
        self.ingester = DocumentIngester(
            self.landmark_hybrid_retriever, 
            self.citing_hybrid_retriever, 
            self.landmark_docstore, 
            self.citing_docstore, 
            child_splitter
        )
        self.retriever = NyayaSetuRetriever(
            self.landmark_hybrid_retriever, 
            self.citing_hybrid_retriever,
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
    def debug_query(self, user_query):
        print("\n" + "🔍" + "="*50)
        print(f"DEBUGGING QUERY: {user_query}")
        
        # 1. Test Citing Retrieval
        c_results, c_scores = self.retriever.two_stage_retrieval(
            user_query, self.citing_hybrid_retriever, k_final=5, return_scores=True
        )
        
        if not c_results:
            print("❌ No Citing chunks found.")
            return

        top_c = c_results[0]
        c_id = top_c.metadata.get('parent_id')
        print(f"✅ Top Citing ID: {c_id} (Score: {c_scores[0]:.4f})")

        # 2. Check Registry
        landmark_ids = self.retriever.relationship_registry.get(c_id)
        print(f"🔗 Registry Match: {landmark_ids}")

        if not landmark_ids:
            print("❌ No landmark mapping found for this Citing ID in relationship_registry.")
            return

        # 3. Test Landmark Retrieval BEFORE filtering
        l_query = self.retriever.generate_landmark_query(user_query, top_c.page_content)
        print(f"📝 Generated Landmark Query: {l_query[:100]}...")
        
        l_raw, l_scores = self.retriever.two_stage_retrieval(
            l_query, self.landmark_hybrid_retriever, k_final=10, return_scores=True
        )

        print(f"\n--- [LANDMARK SEARCH RESULTS (TOP 10)] ---")
        found_target = False
        for i, (doc, score) in enumerate(zip(l_raw, l_scores)):
            p_id = doc.metadata.get('parent_id')
            status = "🎯 MATCH" if p_id in landmark_ids else "❌ WRONG ID"
            if p_id in landmark_ids: found_target = True
            print(f"[{i+1}] Score: {score:.4f} | ID: {p_id} | {status}")
        
        if not found_target:
            print("\n🚨 ISSUE: The landmark you want exists, but didn't rank high enough in the top 10.")
    def debug_retrieval(self, query):
        print("\n" + "="*50)
        print(f"🔍 DEBUGGING RETRIEVAL FOR: '{query}'")
        print("="*50)

        # 1. Check Redis Docstores
        l_keys = list(self.landmark_docstore.yield_keys())
        c_keys = list(self.citing_docstore.yield_keys())
        print(f"📡 Redis Status: {len(l_keys)} Landmarks, {len(c_keys)} Citing docs stored.")

        # 2. Test Citing Retrieval (Stage 1 & 2)
        print("\n--- [STEP 1: CITING RETRIEVAL] ---")
        citing_results, citing_scores = self.retriever.two_stage_retrieval(
            query, self.citing_hybrid_retriever, k_final=3, return_scores=True
        )

        if not citing_results:
            print("❌ No Citing chunks found in Pinecone!")
        else:
            for i, (doc, score) in enumerate(zip(citing_results, citing_scores)):
                p_id = doc.metadata.get('parent_id', 'N/A')
                print(f"[{i+1}] Score: {score:.4f} | Parent: {p_id}")
                print(f"    Snippet: {doc.page_content[:150]}...")

        # 3. Test Landmark Retrieval
        print("\n--- [STEP 2: LANDMARK RETRIEVAL] ---")
        # Let's see if the specific mapping for the top citing result exists
        if citing_results:
            top_p_id = citing_results[0].metadata.get('parent_id')
            mapping = self.retriever.relationship_registry.get(top_p_id)
            print(f"🔗 Registry lookup for '{top_p_id}': {mapping}")
            
            # Generate the specific landmark query the LLM uses
            l_query = self.retriever.generate_landmark_query(query, citing_results[0].page_content)
            print(f"📝 Generated Landmark Query: {l_query[:100]}...")
            
            l_results, l_scores = self.retriever.two_stage_retrieval(
                l_query, self.landmark_hybrid_retriever, k_final=3, return_scores=True
            )
            
            if not l_results:
                print("❌ No Landmark chunks found in Pinecone!")
            else:
                for i, (doc, score) in enumerate(zip(l_results, l_scores)):
                    p_id = doc.metadata.get('parent_id', 'N/A')
                    # Check if this ID is in our allowed registry
                    allowed = "✅ (MATCH)" if (mapping and p_id in mapping) else "⚠️ (NO MATCH)"
                    print(f"[{i+1}] Score: {score:.4f} | Parent: {p_id} {allowed}")
                    print(f"    Snippet: {doc.page_content[:150]}...")
    def debug(self, query):
    # check landmark collection
        keys = list(self.landmark_docstore.yield_keys())
        print("LANDMARK keys in docstore:", keys)
    
    # check citing collection
        keys = list(self.citing_docstore.yield_keys())
        print("CITING keys in docstore:", keys)
    
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

# if __name__ == "__main__":
#     nyayasetu = NyayaSetu()
#     # nyayasetu.ingest("Bangalore_water_supply_1978.pdf", "landmark", "Banglore_water_supply_1978")
#     # nyayasetu.ingest("piyara_Singh.pdf", "landmark", "Piarasingh_Vs_State_Of_Punjab")
#     # nyayasetu.ingest("Secretary_State_Of_Karnataka_And_vs_Umadevi.PDF", "citing", "Secretary_State_Of_Karnataka_And_vs_Umadevi")
#     # response = nyayasetu.query("can contract workers claim permanent employment")
#     # print(response)
#     # nyayasetu.debug("can contract workers claim permanent employment")
#     # nyayasetu.debug("can contract workers claim permanent employment")
#     print("Checking system readiness...")
    
#     # 2. Run the Debug Query
#     test_query = "Can contract workers claim permanent employment?"
#     # nyayasetu.debug_retrieval(test_query)
#     nyayasetu.debug_query(test_query)
    # # 3. Run the Full Pipeline Query
    # print("\n" + "="*50)
    # print("🚀 RUNNING FULL ANALYSIS")
    # print("="*50)
    # final_response = nyayasetu.query(test_query)
    
    # print(f"\nFinal Classification: {final_response['conflict_type']}")
    # print(f"Landmark Case used: {final_response['landmark_case']}")
    # print(f"Citing Case used: {final_response['citing_case']}")
    # print(f"\nAnswer excerpt: {final_response['answer'][:500]}...")
    # eval_questions = [
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
    

    # from ragas import evaluate
    # from ragas.metrics import answer_relevancy, context_precision

# RAGAS Embeddings — reuse your existing embeddings

