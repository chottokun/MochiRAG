# MochiRAG

MochiRAG is a multi-tenant, Retrieval-Augmented Generation (RAG) application that allows users to upload their own documents and interact with an AI to get answers based on the provided content.

This application features a Python backend built with FastAPI and a reactive frontend built with Streamlit.

## ✨ Features

- **Secure Multi-Tenancy**: User data is completely isolated. A user can only access the documents and datasets they own.
- **User Authentication**: Secure sign-up and login functionality.
- **Dataset Management**: Create and delete datasets to organize documents.
- **Document Management**: Upload documents (`.txt`, `.md`, `.pdf`) to specific datasets and delete them.
- **RAG Chat Interface**:
    - Ask questions in natural language.
    - Select one or more datasets to query against.
    - Choose from various RAG strategies for retrieval (see "RAG Strategies" section below).
    - View the sources used by the LLM to generate an answer.

### RAG Strategies

MochiRAG supports multiple RAG (Retrieval-Augmented Generation) strategies, allowing users to experiment with different retrieval approaches. You can select your preferred strategy in the chat interface settings.

- **Basic (Vector Search)**: A standard semantic search based on vector similarity. Documents are split into chunks, embedded, and stored in a vector database. Retrieval involves finding the `k` most similar chunks to the query.
- **Multi-Query Retriever**: Generates multiple variations of the user's question to retrieve a broader set of relevant documents, helping to overcome the limitations of single-query similarity search.
- **Contextual Compression Retriever**: Retrieves a larger set of documents and then uses an LLM to compress and extract only the most relevant information from those documents, reducing noise.
- **Parent Document Retriever**: Stores smaller, highly relevant "child" chunks in the vector database, but retrieves and provides the larger "parent" document to the LLM for context. This helps maintain context while still leveraging precise retrieval.
- **DeepRAG**: A multi-step reasoning strategy that decomposes complex questions into simpler subqueries. It iteratively retrieves information for each subquery and synthesizes the intermediate answers to form a comprehensive final response. The reasoning trace is visible in the UI.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management (recommended).
- [uv](https://astral.sh/uv) as an alternative dependency installer.
- An Ollama instance running with a model (e.g., `gemma3:4b-it-qat`). For installation and usage, refer to the [Ollama documentation](https://ollama.com/). You can pull a model using `ollama pull gemma3:4b-it-qat`. The LLM can be configured in `config/strategies.yaml`.

### 1. Setup

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd MochiRAG
```

**Using Poetry (Recommended):**

```bash
poetry install
```

**Using uv (Alternative):**

First, install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, install the project dependencies:

```bash
uv pip install -e .
```

**Note on Dependencies:** The project dependencies, especially PyTorch and CUDA packages, require a significant amount of disk space (>10 GB). Please ensure you have sufficient space before installation.

### 2. Running the Application

The application consists of two main components: the backend server and the frontend UI. You need to run both in separate terminal sessions.

**Running the Backend:**

The backend is a FastAPI application. Run it using `uvicorn`.

```bash
uvicorn backend.main:app --reload --port 8000
```
The backend server will be available at `http://localhost:8000`.

**Running the Frontend:**

The frontend is a Streamlit application.

```bash
streamlit run frontend/app.py
```
The frontend will be available at `http://localhost:8501`. Open this URL in your browser to use the application.

### Using ChromaDB in Client-Server Mode

By default, MochiRAG runs ChromaDB in a local, file-based mode. For multi-user or larger-scale deployments, you can run ChromaDB as a separate server and configure MochiRAG to connect to it.

**1. Run ChromaDB Server (using Docker):**

The easiest way to run a ChromaDB server is with Docker.

```bash
docker run -p 8000:8000 chromadb/chroma
```

This will start a ChromaDB server listening on `http://localhost:8000`.

**2. Configure MochiRAG:**

Next, update the `config/strategies.yaml` file to tell MochiRAG to connect to the Chroma server instead of running its own local instance.

```yaml
# config/strategies.yaml

vector_store:
  provider: chromadb
  mode: http       # Change 'persistent' to 'http'
  host: localhost  # Or the IP of your Docker host
  port: 8000
  # The 'path' property is ignored in http mode
```

After making this change, restart the MochiRAG backend. It will now connect to the external ChromaDB server.

## 🧪 Running Tests

The test suite is built with `pytest` and is designed to run without any external service dependencies (it uses an in-memory SQLite database and mocks for API tests).

To run the tests, execute the following command from the root of the project directory:

```bash
pytest
```

This will discover and run all tests in the `tests/` directory.

## 📂 Project Structure

- `backend/`: FastAPI application for API endpoints, authentication, and data management.
- `core/`: Core logic for RAG functionalities, including LLM interaction, embedding, retrieval, and vector store management.
- `frontend/`: Streamlit application for the user interface.
- `config/`: Configuration files, such as RAG strategies.
- `tests/`: Unit and integration tests for the backend and core modules.
- `docs/`: Project documentation, requirements, and design documents.
