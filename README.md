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
    - Choose the RAG strategy to be used for retrieval (currently supports `basic` vector search).
    - View the sources used by the LLM to generate an answer.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management.
- An Ollama instance running with a model (e.g., `gemma3:4b-it-qat`). The LLM can be configured in `config/strategies.yaml`.

### 1. Setup

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd MochiRAG
pip install -e .
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

## 🧪 Running Tests

The test suite is built with `pytest` and is designed to run without any external service dependencies (it uses an in-memory SQLite database and mocks for API tests).

To run the tests, execute the following command from the root of the project directory:

```bash
pytest
```

This will discover and run all tests in the `tests/` directory.
