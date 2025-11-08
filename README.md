# MochiRAG

MochiRAG is a multi-tenant, feature-rich Retrieval-Augmented Generation (RAG) application. It provides a secure environment for users to upload documents and interact with a Large Language Model (LLM) to get answers based on their own content.

![MochiRAG System Architecture](docs/architecture/system_architecture.png)
*(A placeholder for the architecture diagram which can be generated from the mermaid code in `docs/architecture/system_architecture.md`)*

## ✨ Key Features

- **Secure Multi-Tenancy**: User data is completely isolated. A user can only access the documents and datasets they own.
- **Pluggable RAG Strategies**: Go beyond simple vector search. MochiRAG supports multiple advanced RAG strategies out-of-the-box, including `MultiQuery`, `ParentDocument`, and `StepBackPrompting`.
- **Advanced Document Processing**: Utilizes the `Docling` library for robust parsing of PDFs, including complex layouts, tables, and OCR for scanned documents.
- **Configurable Ingestion**: Control document chunk size and overlap directly from your `.env` file for fine-tuned performance.
- **Flexible LLM Selection**: Easily switch between LLM providers (Ollama, OpenAI, Azure, Gemini) and models via environment variables.
- **Extensible Framework**: Easily add new RAG strategies and LLM providers through simple configuration changes.
- **Shared Knowledge Bases**: Administrators can create read-only "shared databases" accessible to all users, perfect for common knowledge bases.
- **Developer-Friendly**: Comes with a comprehensive test suite, a `docker-compose` setup for easy local development, and extensive documentation.

## 🛠️ Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Frontend**: Streamlit
- **Core RAG**: LangChain
- **Vector Store**: ChromaDB
- **Dependency Management**: Poetry

## 🚀 Getting Started (The Easy Way)

The fastest way to get the entire MochiRAG application running on your local machine is with `docker-compose`.

### Prerequisites

- Docker and `docker-compose` are installed.
- An LLM running locally (e.g., via [Ollama](https://ollama.com/)).

### 1. Configure the Application

First, create a `.env` file in the project root. You can copy the example to get started:

```bash
cp .env.example .env
```

Make sure the `SECRET_KEY` is set to a long, random string. If your local LLM (Ollama) is running, the default settings should work.

### 2. Launch the Application

Run the following command from the project root:

```bash
docker-compose up --build
```

This will build the images and start all the required services: the backend, the frontend, and the ChromaDB vector store.

Once startup is complete, you can access the application:
- **Frontend UI**: `http://localhost:8501`
- **Backend API Docs**: `http://localhost:8000/docs`

## 📚 Documentation

This repository contains extensive documentation covering architecture, design decisions, and guides for developers.

**[➡️ Explore the Full Documentation in the `docs` Directory](./docs)**

Key documents include:
- **[System Architecture](./docs/architecture/system_architecture.md)**: A high-level overview of the components.
- **[Implementation Design](./docs/architecture/implementation_design.md)**: A deep dive into the `LLMManager` and `RetrieverManager`, explaining each RAG strategy in detail.
- **[Developer Setup Guide](./docs/guides/developer_setup_guide.md)**: Instructions for a manual, non-Docker setup.
- **[Deployment Guide](./docs/guides/DEPLOY.md)**: How to build and push Docker images for production.

## 💻 Development and Testing

For developers who want to run the services manually without Docker:

1.  **Install Dependencies**: `poetry install`
2.  **Run Backend**: `uvicorn backend.main:app --reload --port 8000`
3.  **Run Frontend**: `streamlit run frontend/app.py`

To run the test suite, use `pytest`:
```bash
pytest
```
For more details, see the [Developer Setup Guide](./docs/guides/developer_setup_guide.md).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
