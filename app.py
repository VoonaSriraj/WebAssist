import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="ChatGroq RAG Demo",
    page_icon="🤖",
    layout="wide"
)

# Load the Groq API key
groq_api_key = os.environ.get('GROQ_API_KEY')

if not groq_api_key:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

# Initialize vector store (only once)
@st.cache_resource
def initialize_vector_store():
    try:
        with st.spinner("Loading and processing documents..."):
            # Use HuggingFace embeddings (more reliable)
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            loader = WebBaseLoader("https://www.geeksforgeeks.org/c-plus-plus/")
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs[:50])
            vectors = FAISS.from_documents(final_documents, embeddings)
            
        return vectors, embeddings
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None, None

# Initialize components
vectors, embeddings = initialize_vector_store()

if vectors is None:
    st.stop()

# Streamlit UI
st.title("🤖 ChatGroq RAG Demo")
st.write("Ask questions about LangSmith documentation!")

# Initialize LLM
try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",  # Updated to current supported model
        temperature=0
    )
except Exception as e:
    st.error(f"Error initializing ChatGroq: {str(e)}")
    st.stop()

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    If you cannot find the answer in the context, say "I don't have enough information to answer this question."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Answer:
    """
)

# Create chains
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever(search_kwargs={"k": 5})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input
user_prompt = st.text_input("💬 Enter your question here:")

if user_prompt:
    try:
        with st.spinner("Generating response..."):
            start_time = time.time()
            response = retrieval_chain.invoke({"input": user_prompt})
            response_time = time.time() - start_time
        
        # Display response
        st.success("✅ Response generated!")
        st.write("**Answer:**")
        st.write(response['answer'])
        
        # Show response time
        st.info(f"⏱️ Response time: {response_time:.2f} seconds")
        
        # Document similarity search results
        with st.expander("📄 View Source Documents"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
                
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app demonstrates RAG using:")
    st.write("- **ChatGroq** for LLM inference")
    st.write("- **LangChain** for document processing")
    st.write("- **FAISS** for vector storage")
    st.write("- **Ollama** for embeddings")
    
    st.header("🔧 Configuration")
    st.write(f"**Model:** mixtral-8x7b-32768")
    st.write(f"**Embedding Model:** nomic-embed-text")
    st.write(f"**Vector Store:** FAISS")