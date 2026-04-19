from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
from dotenv import load_dotenv

loader = PyPDFLoader("_attention_is_all_you_need_the_paper_that_changed_ai.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=45)
chunks = splitter.split_documents(docs)
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
import os

if not os.path.exists("chromadb"):
    vectorStore = Chroma(
        collection_name="sample",
        persist_directory="chromadb",
        embedding_function=embedding
    )
    vectorStore.add_documents(chunks)
else:
    vectorStore = Chroma(
        collection_name="sample",
        persist_directory="chromadb",
        embedding_function=embedding
    )
result = vectorStore.similarity_search(query='''What problem were the authors trying to solve?"
"What is multi head attention?"  
"Why did they remove recurrence from the architecture?''',k=5)
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
