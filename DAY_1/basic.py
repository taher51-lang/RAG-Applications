from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
chat_model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.8
)
loader = PyPDFLoader("100DaysofML_Complete_Course_CampusX.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = []
for doc in loader.lazy_load():
    split = splitter.split_documents([doc])
    chunks.extend(split)
    print(f"Processed page {doc.metadata['page'] + 1}")

print(f"Total chunks: {len(chunks)}")
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
with tempfile.TemporaryDirectory() as temp:
    vectorStore = Chroma(
        collection_name="sample",
        persist_directory="chromaDB",
        embedding_function=embedding
    )
    vectorStore.add_documents(chunks)
if not os.path.exists("chromadb"):
    vectorStore = Chroma(
        collection_name="sample",
        persist_directory="chromadb",
        embedding_function=embedding
    )
    # vectorStore.add_documents(chunks)
else:
    vectorStore = Chroma(
        collection_name="sample",
        persist_directory="chromadb",
        embedding_function=embedding
    )
result = vectorStore.similarity_search(query='''what is Machine learning''',k=20)
prompt = PromptTemplate(
    template="From the given context : {context} , answer the given question, if relevant info not found then return no info found , questions : {question}",
    input_variables=['context','question']
)
query = '''explain mathematical transformation'''
# What is multi head attention?Why did they remove recurrence from the architecture?'''
chain = prompt | chat_model | StrOutputParser()
context = context = "\n\n".join([r.page_content for r in result])
print(context)
print("--------")
output = chain.invoke({"context":context,"question":query})
print(output)
