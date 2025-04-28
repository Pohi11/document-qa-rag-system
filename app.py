import streamlit as st
import os
import shutil
import time
import gc
import chromadb
from chromadb.config import Settings

# Import LangChain components
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# --- Constants ---
UPLOAD_DIR = "./temp_uploads"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

# --- Global ChromaDB Settings ---
CHROMA_SETTINGS = Settings(
    allow_reset=True,
    anonymized_telemetry=False
)

# --- Cached Resource Functions ---
@st.cache_resource
def get_chroma_client():
    """Gets a singleton ChromaDB PersistentClient instance with reset enabled."""
    print("Initializing ChromaDB PersistentClient...")
    return chromadb.PersistentClient(path=PERSIST_DIR, settings=CHROMA_SETTINGS)

@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace embedding model."""
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

@st.cache_resource
def load_llm():
    """Loads the Ollama LLM."""
    print("Loading LLM...")
    return ChatOllama(model=LLM_MODEL_NAME, temperature=0.2)

# --- Database Management Functions ---
def reset_chroma_database(client):
    """Resets the Chroma database using the provided client instance."""
    print(f"Attempting to reset Chroma database via client...")
    try:
        # First reset the client
        client.reset()
        print(f"Chroma database reset successfully via client.")

        # Force close any existing collections
        for collection in client.list_collections():
            try:
                client.delete_collection(collection.name)
                print(f"Deleted collection: {collection.name}")
            except Exception as e:
                print(f"Error deleting collection {collection.name}: {e}")

        # Ensure the directory is completely clean
        if os.path.exists(PERSIST_DIR):
            try:
                # Remove all files in the directory
                for filename in os.listdir(PERSIST_DIR):
                    file_path = os.path.join(PERSIST_DIR, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        print(f"Removed: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
                
                # Remove the directory itself and recreate it
                shutil.rmtree(PERSIST_DIR)
                os.makedirs(PERSIST_DIR)
                print(f"Recreated clean directory {PERSIST_DIR}")
            except Exception as e:
                print(f"Error cleaning directory: {e}")

        return True
    except Exception as e:
        st.error(f"An error occurred while resetting the Chroma database: {e}")
        print(f"Error during ChromaDB reset: {e}")
        return False

def load_vector_store(embeddings):
    """Loads an existing LangChain Chroma vector store."""
    print(f"Attempting to load vector store from {PERSIST_DIR}")
    
    # Additional check for truly empty database
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print("Persistence directory doesn't exist or is empty.")
        return None
        
    try:
        # Check if the directory contains actual Chroma files
        required_files = ['chroma.sqlite3']
        existing_files = os.listdir(PERSIST_DIR)
        if not any(f for f in required_files if f in existing_files):
            print("No valid Chroma database files found.")
            return None

        # Try to load the vector store
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        
        # Verify the store has actual content
        try:
            count = vector_store._collection.count()
            print(f"Vector store count check: {count}")
            if count == 0:
                print("Vector store exists but is empty.")
                return None
            print(f"Successfully loaded vector store with {count} entries.")
            return vector_store
        except Exception as count_e:
            print(f"Error checking vector store count: {count_e}")
            return None
            
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def create_vector_store(chunks, embeddings):
    """Creates a new Chroma vector store from document chunks."""
    print(f"Creating new vector store at {PERSIST_DIR} with {len(chunks)} chunks...")
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            client_settings=CHROMA_SETTINGS  # Use the same settings
        )
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        print(f"Error creating vector store: {e}")
        return None

def process_documents(uploaded_files, embeddings):
    """Loads, splits documents and returns chunks."""
    print("Processing uploaded documents (loading & splitting)...")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    all_chunks = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            continue

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"Processed {len(documents)} pages from {uploaded_file.name} into {len(chunks)} chunks.")

    # Cleanup uploads directory
    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
            print("Temporary upload directory cleaned up.")
        except Exception as e:
            st.warning(f"Could not clean up temporary upload directory: {e}")

    if not all_chunks:
        st.warning("No processable documents found or failed to process.")
        return None

    print(f"Returning {len(all_chunks)} chunks for vector store creation.")
    return all_chunks

def setup_qa_chain(vector_store, llm):
    """Sets up the RetrievalQA chain."""
    print("Setting up QA chain...")
    template = """You are an assistant specialized in answering questions based ONLY on the provided context document excerpts.
    Use the following pieces of retrieved context to answer the question.
    If the answer is not found within the context, state clearly "The answer is not found in the provided documents." Do not make up information.
    Keep your answer concise and directly address the question. Use a maximum of three sentences.

    Context: {context}

    Question: {question}

    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("QA Chain ready.")
    return qa_chain

# --- Streamlit Application ---
st.set_page_config(page_title="Doc Q&A with RAG", layout="wide")
st.title("📄 Document Question Answering with RAG")

# --- Sidebar ---
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents:", type="pdf", accept_multiple_files=True
)

process_button_clicked = st.sidebar.button("Process Uploaded Documents")

st.sidebar.markdown("---")
st.sidebar.header("2. Database Management")
reset_button_clicked = st.sidebar.button("⚠️ Reset Database (Clear All Data)")
st.sidebar.markdown("*(Use this if you want to start fresh or encounter issues)*")

st.sidebar.markdown("---")
st.sidebar.markdown(f"Using LLM: `{LLM_MODEL_NAME}`")
st.sidebar.markdown("*(Ensure Ollama is running)*")

# --- Load Foundational Models & Client ---
embeddings = load_embedding_model()
llm = load_llm()
client = get_chroma_client()  # Get the singleton client

# --- Session State Initialization ---
default_session_state = {
    "qa_chain": None,
    "db_ready": False,
    "processed_this_session": False
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- App Logic ---
# Handle Reset First
if reset_button_clicked:
    with st.spinner("Resetting database..."):
        # Clear all session state
        st.session_state.qa_chain = None
        st.session_state.db_ready = False
        st.session_state.processed_this_session = False
        
        # Force garbage collection
        gc.collect()
        
        # Get a fresh client instance
        try:
            # Clear the cached client
            get_chroma_client.clear()
            client = get_chroma_client()
            
            # Reset the database
            reset_success = reset_chroma_database(client)
            
            if reset_success:
                st.sidebar.success("Database reset successfully!")
                st.info("Database has been reset. Please upload and process new documents.")
                
                # Clear the cache again to ensure fresh start
                get_chroma_client.clear()
                # Force reload of the page to ensure clean state
                time.sleep(0.5)  # Small delay to ensure file operations complete
                st.rerun()
            else:
                st.sidebar.error("Database reset failed. Check error messages.")
                st.error("Please try manually deleting the chroma_db folder and restart the application.")
        except Exception as e:
            st.error(f"Error during reset process: {e}")
            print(f"Reset error details: {e}")
    st.stop()  # Stop execution after reset attempt

# Handle Processing
elif process_button_clicked:
    if uploaded_files:
        with st.spinner("Processing documents... This may take a while."):
            chunks = process_documents(uploaded_files, embeddings)

            if chunks:
                vector_store_instance = create_vector_store(chunks, embeddings)

                if vector_store_instance:
                    st.session_state.qa_chain = setup_qa_chain(vector_store_instance, llm)
                    st.session_state.db_ready = True
                    st.session_state.processed_this_session = True
                    st.sidebar.success("Documents processed successfully!")
                else:
                    st.sidebar.error("Failed to create vector store after processing.")
                    st.session_state.db_ready = False
            else:
                st.sidebar.error("Document processing failed (no chunks generated).")
                st.session_state.db_ready = False
    else:
        st.sidebar.warning("Please upload at least one PDF document.")
    st.rerun()

# Attempt to load existing state
elif not st.session_state.db_ready:
    print("Attempting to load existing vector store on app load/refresh...")
    vector_store_instance = load_vector_store(embeddings)
    if vector_store_instance:
        st.session_state.qa_chain = setup_qa_chain(vector_store_instance, llm)
        st.session_state.db_ready = True
        print("Existing database loaded and QA chain ready.")
    else:
        print("No existing, valid database found to load.")
        st.session_state.db_ready = False

# --- Display Status & Q&A Section ---
if st.session_state.db_ready:
    if not st.session_state.processed_this_session:
        st.info("Database ready. Ask questions or upload new documents (this will replace the current data).")
elif not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    st.warning("Please upload and process documents to enable Q&A.")

st.header("3. Ask Questions")
query = st.text_input(
    "Enter your question about the documents:",
    disabled=not st.session_state.db_ready
)

if query and st.session_state.qa_chain:
    with st.spinner("Searching documents and generating answer..."):
        try:
            result = st.session_state.qa_chain.invoke({"query": query})
            st.subheader("Answer:")
            st.write(result["result"])
            st.subheader("Sources:")
            source_docs = result.get("source_documents", [])
            if source_docs:
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get('source', 'Unknown Source')
                    page = doc.metadata.get('page', '?')
                    st.write(f"**Source {i+1} (Page {page+1} of {os.path.basename(source)}):**")
                    st.write(f"> {doc.page_content[:250]}...")
                    st.divider()
            else:
                st.write("No specific source documents were identified for this answer.")
        except Exception as e:
            st.error(f"An error occurred during question answering: {e}")
elif query:
    st.warning("The QA system is not ready. Please process documents first.")

# Reset the 'processed_this_session' flag at the end
st.session_state.processed_this_session = False