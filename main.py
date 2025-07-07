import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit page title
st.title(" PDF Chatbot with RAG (Streamlit + LangChain)")

# Load and embed PDF - Cache so it runs only once
@st.cache_resource
def load_vector_store():
    # 1 Load the PDF
    loader = PyPDFLoader("index_card.pdf")
    docs = loader.load()

    # 2 Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3 Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

    return vector_store

# Initialize vector store
vector_store = load_vector_store()

# Set up the LLM and RetrievalQA chain
llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# User input
user_question = st.text_input("Ask a question about the PDF:")

# Generate and show answer
if user_question:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_question)
    st.success(answer)
