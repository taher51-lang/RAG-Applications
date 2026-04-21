from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import tempfile
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
chat_model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.8
)
loader = PyPDFLoader("_attention_is_all_you_need_the_paper_that_changed_ai.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=45)
chunks = splitter.split_documents(docs)
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
with tempfile.TemporaryDirectory() as temp:
    vectorStore = Chroma(
        collection_name="sample",
        persist_directory=temp,
        embedding_function=embedding
    )
    vectorStore.add_documents(chunks)
    result = vectorStore.similarity_search(query='''What problem were the authors trying to solve?"
"What is multi head attention?"  
"Why did they remove recud rrence from the architecture?''',k=5)
prompt = PromptTemplate(
    template="From the given context : {context} , answer the given question, if relevant info not found then return no info found , questions : {question}",
    input_variables=['context','question']
)
query = '''What problem were the authors trying to solve?
What is multi head attention?Why did they remove recurrence from the architecture?'''
chain = prompt | chat_model | StrOutputParser()
context = context = "\n\n".join([r.page_content for r in result])
print(context)
print("--------")
output = chain.invoke({"context":context,"question":query})
print(output)
