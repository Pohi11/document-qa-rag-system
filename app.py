import streamlit as st
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# --- Constants ---
# Use a temporary directory for uploaded files during processing
UPLOAD_DIR = "./temp_uploads"
# Directory where the Chroma vector database will be persisted
PERSIST_DIR = "./chroma_db"
# Embedding model name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Ollama LLM model name (make sure you have pulled this with 'ollama pull ...')
LLM_MODEL_NAME = "llama3.2"

# --- Helper Functions ---

# Function to load embedding model - cached for efficiency
@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace embedding model."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if GPU is available
        encode_kwargs={'normalize_embeddings': False}
    )
    print("Embedding model loaded.")
    return embeddings

# Function to load LLM - cached for efficiency
@st.cache_resource
def load_llm():
    """Loads the Ollama LLM."""
    print("Loading LLM...")
    # Ensure Ollama service is running
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.2) # Low temperature for factual answers
    print("LLM loaded.")
    return llm

# Function to process uploaded documents (load, split, embed, store)
def process_documents(uploaded_files, embeddings):
    """Loads, splits, embeds, and stores documents in ChromaDB."""
    print("Processing uploaded documents...")
    # Create a temporary directory for uploads if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    all_chunks = []
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the PDF
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load() # Each page is a document
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            continue

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"Processed {len(documents)} pages from {uploaded_file.name} into {len(chunks)} chunks.")

    if not all_chunks:
        st.warning("No processable documents found or failed to process.")
        # Clean up temp directory if empty or failed
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        return None

    # Create or update Chroma vector store
    print(f"Creating/updating vector store with {len(all_chunks)} chunks...")
    # Clear old database directory if it exists before creating a new one
    if os.path.exists(PERSIST_DIR):
        print(f"Removing existing vector store at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"Vector store created at {PERSIST_DIR}")

    # Clean up the temporary upload directory
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    print("Temporary upload directory cleaned up.")

    return vector_store


# Function to setup the QA chain
def setup_qa_chain(vector_store, llm):
    """Sets up the RetrievalQA chain."""
    print("Setting up QA chain...")
    # Define the prompt template
    template = """You are an assistant specialized in answering questions based ONLY on the provided context document excerpts.
    Use the following pieces of retrieved context to answer the question.
    If the answer is not found within the context, state clearly "The answer is not found in the provided documents." Do not make up information.
    Keep your answer concise and directly address the question. Use a maximum of three sentences.

    Context: {context}

    Question: {question}

    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Create a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Use similarity search
        search_kwargs={'k': 3}    # Retrieve top 3 relevant chunks
    )

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' puts all retrieved text directly into the prompt
        retriever=retriever,
        return_source_documents=True, # Include source document info
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("QA Chain ready.")
    return qa_chain

# --- Streamlit Application ---

st.set_page_config(page_title="Doc Q&A with RAG", layout="wide")
st.title("📄 Document Question Answering with RAG")

st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF documents here:",
    type="pdf",
    accept_multiple_files=True
)

# Load models (cached)
embeddings = load_embedding_model()
llm = load_llm()

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Button to process documents
if st.sidebar.button("Process Uploaded Documents"):
    if uploaded_files:
        with st.spinner("Processing documents... This may take a while depending on size and number of files."):
            st.session_state.vector_store = process_documents(uploaded_files, embeddings)
            if st.session_state.vector_store:
                st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store, llm)
                st.session_state.processing_done = True
                st.sidebar.success("Documents processed successfully!")
            else:
                st.sidebar.error("Document processing failed.")
                st.session_state.processing_done = False
    else:
        st.sidebar.warning("Please upload at least one PDF document.")

# Display information about processed state
if st.session_state.processing_done:
    st.info("Documents processed. You can now ask questions.")
elif os.path.exists(PERSIST_DIR):
     st.info("Existing vector store found. Loading QA chain. If you want to use new documents, please upload and process them.")
     # Try to load existing store if processing wasn't done this session
     if st.session_state.vector_store is None:
         try:
             st.session_state.vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
             st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store, llm)
             st.session_state.processing_done = True # Mark as ready
             print("Successfully loaded existing vector store.")
         except Exception as e:
             st.error(f"Failed to load existing vector store: {e}. Please upload and process documents.")
             st.session_state.processing_done = False

else:
    st.warning("Please upload and process documents to enable Q&A.")


st.header("2. Ask Questions")
# Disable input if documents haven't been processed
query = st.text_input(
    "Enter your question about the documents:",
    disabled=not st.session_state.processing_done
)

if query and st.session_state.qa_chain:
    with st.spinner("Searching documents and generating answer..."):
        try:
            result = st.session_state.qa_chain.invoke({"query": query})

            st.subheader("Answer:")
            st.write(result["result"])

            st.subheader("Sources:")
            # Display source documents used for the answer
            source_docs = result.get("source_documents", [])
            if source_docs:
                for i, doc in enumerate(source_docs):
                    # Access metadata for source and page number
                    source = doc.metadata.get('source', 'Unknown Source')
                    page = doc.metadata.get('page', '?')
                    st.write(f"**Source {i+1} (Page {page+1} of {os.path.basename(source)}):**") # Page numbers are often 0-indexed
                    # Display a snippet of the source text
                    st.write(f"> {doc.page_content[:250]}...") # Show first 250 chars
                    st.divider()
            else:
                st.write("No specific source documents were identified for this answer.")

        except Exception as e:
            st.error(f"An error occurred during question answering: {e}")

elif query:
    st.warning("The QA system is not ready. Have you processed the documents?")

# Add a footer or instructions
st.sidebar.markdown("---")
st.sidebar.markdown("Ensure Ollama is running with the model `llama3.2` available.")
st.sidebar.markdown("Processing large documents can take time.")