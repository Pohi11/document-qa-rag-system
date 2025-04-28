# Generative AI Document Q&A (RAG) with Streamlit & Ollama

## Overview

This project implements a Question Answering system using the Retrieval-Augmented Generation (RAG) pattern. It allows users to upload PDF documents and ask natural language questions about their content. The system retrieves relevant passages from the documents and uses a local Large Language Model (LLM) via Ollama to generate answers based *only* on the provided context.

This application was developed primarily as a learning exercise to gain hands-on experience with key concepts and technologies in the field of Generative AI, including:
*   Large Language Models (LLMs)
*   Retrieval-Augmented Generation (RAG) pipelines
*   Vector Embeddings and Vector Stores
*   Text Chunking Strategies
*   Integrating these components into a functional application.

The goal is to provide answers grounded in the uploaded documents, mitigating hallucination and providing source context for verification.

## Key Features

*   **PDF Document Upload:** Supports uploading multiple PDF files via a user-friendly interface.
*   **Document Processing:** Automatically loads, parses, and splits documents into manageable chunks.
*   **Local Embeddings:** Uses a sentence-transformer model (`all-MiniLM-L6-v2`) running locally to generate vector embeddings for document chunks.
*   **Local Vector Store:** Stores embeddings and document chunks in a local ChromaDB database for efficient retrieval, persisting across sessions.
*   **Local LLM Integration:** Leverages Ollama to run powerful open-source LLMs (specifically `llama3.2` in this configuration) locally for answer generation.
*   **RAG Pipeline:** Implements the core RAG logic using LangChain:
    *   Embeds the user's question.
    *   Retrieves relevant document chunks from ChromaDB based on semantic similarity.
    *   Constructs a prompt containing the retrieved context and the original question.
    *   Feeds the prompt to the local LLM to generate an answer.
*   **Context-Grounded Answers:** The LLM is specifically prompted to answer *only* based on the retrieved context.
*   **Source Display:** Shows snippets from the source document chunks used to generate the answer.
*   **Web Interface:** Built with Streamlit for easy interaction.

## Technology Stack

*   **Core Logic:** Python 3.8+
*   **GenAI/NLP Framework:** LangChain
*   **Web UI:** Streamlit
*   **LLM Runner:** Ollama
*   **LLM Model:** Llama 3.2 (via Ollama tag, e.g., `llama3.2`)
*   **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`)
*   **Vector Database:** ChromaDB
*   **PDF Handling:** PyPDFLoader (via LangChain)

## Setup and Installation

**Prerequisites:**

1.  **Python:** Ensure you have Python 3.8 or newer installed.
2.  **Pip:** Python's package installer, usually included with Python.
3.  **Ollama:** Install Ollama for your operating system from [https://ollama.com/](https://ollama.com/).
4.  **Ollama Model:** Run the required LLM model. Open your terminal and run:
    ollama run llama3.2
5.  **Ollama Running:** Make sure the Ollama application/service is running in the background before starting the Streamlit app. You can test this by running `ollama list` in your terminal.

**Installation Steps:**

1.  **Clone the Repository (Optional):** If you have this project in a Git repository:
    ```bash
    git clone <your-repository-url>
    cd rag_qa_project
    ```
    If not, simply navigate to your project directory (`rag_qa_project`).

2.  **Create and Activate Virtual Environment:**
    # Create the virtual environment
    python -m venv venv

    # Activate it (choose the command for your OS)
    # Windows:
    ```bash
    .\venv\Scripts\activate
    ```
    # macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

3.  **Install Dependencies:** Create a `requirements.txt` file (as provided separately) in your project directory, then run:
```bash
    pip install -r requirements.txt
```

4.  **Verify Ollama is Ready:** Run `ollama list` in your terminal again to ensure the `llama3.2` model (or the correct tag) is listed and the service is accessible.

## Usage

1.  **Ensure Ollama is Running:** Double-check that the Ollama application or service is active with the `llama3.2` model available.
2.  **Navigate to Project Directory:** Open your terminal in the `rag_qa_project` directory where `app.py` is located.
3.  **Activate Virtual Environment:** If not already active, run `source venv/bin/activate` (or the Windows equivalent).
4.  **Run the Streamlit App:**
    streamlit run app.py
5.  **Interact with the App:**
    *   Your web browser should open automatically to the application (usually `http://localhost:8501`).
    *   Use the sidebar ("1. Upload Documents") to upload one or more PDF files.
    *   Click the "Process Uploaded Documents" button in the sidebar. Wait for the processing spinner to finish. A success message will appear in the sidebar.
    *   Once processed, type your question about the content of the uploaded documents into the main text input area ("2. Ask Questions") and press Enter.
    *   The generated answer will appear, followed by snippets from the source documents that were used as context.

## Project Structure

```text
document-qa-system/
│
├── venv/                  # Python virtual environment (not committed)
├── chroma_db/             # Persisted ChromaDB vector store (created on first run)
├── temp_uploads/          # Temporary storage for uploads during processing (deleted after)
├── app.py                 # Main Streamlit application script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Potential Improvements & Future Work

*   **Error Handling:** Implement more robust error handling for file processing and API calls.
*   **Document Types:** Add support for other document formats (e.g., DOCX, TXT).
*   **Advanced Chunking:** Experiment with semantic chunking or other strategies.
*   **Model Selection:** Allow users to select different embedding or LLM models available via Ollama.
*   **Vector Store Updates:** Implement logic to add/update documents in the vector store without full replacement.
*   **Asynchronous Processing:** Make document processing asynchronous to prevent UI blocking.
*   **UI Enhancements:** Improve UI with loading indicators, document metadata display, chat history, etc.
*   **Containerization:** Create a `Dockerfile` to easily package and deploy the application.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details (if applicable).