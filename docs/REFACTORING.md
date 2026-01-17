# Refactoring Documentation

This document outlines the comprehensive refactoring performed on MochiRAG to improve maintainability, scalability, and code quality.

## 1. Backend Architecture Overhaul

The backend has been transitioned from a monolithic `main.py` to a modular structure using FastAPI routers.

### New Structure
- `backend/main.py`: Entry point, initializes the FastAPI application, includes routers, and defines global exception handlers.
- `backend/routers/`:
    - `auth.py`: Authentication and token generation.
    - `users.py`: User registration and management.
    - `datasets.py`: Dataset CRUD operations.
    - `documents.py`: Document upload and ingestion management.
    - `chat.py`: Q&A and RAG strategy execution.
- `backend/dependencies.py`: Shared FastAPI dependencies like `get_db` and `get_current_user`.

### Benefits
- **Clear Separation of Concerns**: Each module handles a specific domain.
- **Improved Testability**: Individual routers can be tested in isolation.
- **Easier Navigation**: Developers can quickly find relevant code.

## 2. Configuration Management

The `ConfigManager` was refactored to use `pydantic-settings` (v2), providing a more robust and type-safe way to handle environment variables and secrets.

### Key Changes
- Introduced a `Settings` class that automatically loads from `.env`.
- Centralized all environment-based configuration (secrets, hostnames, ports).
- Maintained support for YAML-based strategy configuration while allowing environment overrides.

## 3. Core Service Improvements

### Standardized Logging
- Replaced all `print` statements with the standard `logging` module.
- Configured a consistent logging format in `backend/main.py`.

### Prompt Externalization
- Hardcoded prompt templates were moved from `core/rag_chain_service.py` and other services to a dedicated `core/prompts.py` file.
- This allows for easier tuning and localization of prompts without touching the logic.

### Service Refinement
- `LLMManager`, `RetrieverManager`, `IngestionService`, and `VectorStoreManager` now use centralized logging and have improved error handling.
- `LLMManager` now logs cache hits/misses and provider instantiation.

## 4. Enhanced Error Handling

- Implemented a **Global Exception Handler** in `backend/main.py` that catches unhandled exceptions, logs them with a traceback, and returns a consistent `500 Internal Server Error` response to the client.
- Improved validation error reporting through Pydantic's built-in features.

## 5. Testing and Verification

- **API Integration Tests**: Added a new test suite (`tests/backend/test_api_v2.py`) that covers the entire user lifecycle.
- **Configuration Tests**: Added `tests/core/test_config_manager_v2.py` to verify the new settings logic and environment variable overrides.
- **Core Service Unit Tests**: Added `tests/core/test_ingestion_service_v2.py` and `tests/core/test_retriever_manager_v2.py` to verify retry logic and strategy selection.
- **Regression Testing**: All existing tests were updated to work with the new structure and verified to pass.

## 6. How to Run Tests

Ensure you have a `.env` file with at least a `SECRET_KEY` set.

```bash
export SECRET_KEY=your_secret_key
poetry run pytest
```
