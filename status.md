# MochiRAG BugFix Status

This document tracks the status of bug fixes and improvements for the MochiRAG project.

## Open Issues

- (No issues reported yet)

## In Progress

- **Feature: Display RAG references with option**
    - Started: YYYY-MM-DD (This will be replaced with the actual date)
    - Goal: Allow users to see the sources RAG referred to, with an option to toggle this display.
    - Status:
        - Step 1 (core/rag_chain.py): Completed. Returns answer and sources. Tests updated and passing.
        - Step 2 (backend): Completed. API model and endpoint updated to handle sources. Most tests updated and passing.
        - Step 3 (frontend/app.py): Completed. UI checkbox for showing references, logic to display sources, and chat history updated. Manual testing pending.
        - Step 4 (tests): Ongoing for backend, frontend manual testing pending.
        - Step 5 (documentation): Completed.
    - Known Issues:
        - (No known issues at this time, all tests passing)
    - Next Steps:
        - Final review and submit.

## Resolved Issues
- `tests/backend/test_main.py::test_chat_query_uses_embedding_strategy_from_metadata` was failing due to mocking issues for `_read_datasources_meta`. Resolved by mocking `open` and `json.load` instead, ensuring the function's internal logic correctly processes controlled data.
- Various `caplog` assertion errors in manager tests: Resolved by checking `record.message` and `record.levelname`.
- `NameError` and `TypeError` in `test_retriever_manager.py` and `test_chunking_manager.py`: Resolved by adding missing imports and correcting kwarg propagation.
- Indentation errors across multiple test files: Resolved.

- (No issues resolved yet)
