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
        - Implement CI/CD pipeline improvements.
        - Enhance tests with more parametrization.

## In Progress (New Plan: Test Infrastructure Improvement)

- **Task: CI/CD Pipeline Integration/Enhancement**
    - Started: YYYY-MM-DD (To be updated)
    - Goal: Automate test execution on code changes to ensure continuous quality.
    - Current Sub-task: Initial GitHub Actions workflow file created. (Verification pending actual push)
    - Plan:
        1. Current setup investigation. (Completed: No existing CI/CD config found)
        2. Platform selection/configuration (GitHub Actions preferred if none exists). (Selected: GitHub Actions)
        3. Test execution job creation (Python setup, dependencies, pytest execution). (Completed: Basic workflow file `.github/workflows/python-app.yml` created)
        4. Trigger configuration (on push/PR to main/develop branches). (Defined in workflow)
        5. Result notification setup. (Basic pass/fail via GitHub Checks)
        6. Operation verification and adjustment. (Pending: Requires push to GitHub and observation)

- **Task: Parameterized Test Enhancement**
    - Goal: Improve test readability, maintainability, and coverage.
    - Current Sub-task: Applying parametrization to `tests/core/test_chunking_manager.py`. (Partially completed, further review/extension pending)
    - Plan:
        1. Identify applicable test areas (manager classes, API endpoint validation). (Ongoing - ChunkingManager identified)
        2. Refactor existing tests using `@pytest.mark.parametrize`. (Partially completed for ChunkingManager)
        3. Add new test cases for better coverage of variations.
        4. Review and merge.
        5. Extend to other manager tests (EmbeddingManager, RetrieverManager) and backend API tests.

- **Task: Test Performance Optimization (Addressing Timeouts)**
    - Started: YYYY-MM-DD (To be updated)
    - Goal: Reduce test suite execution time and eliminate timeouts.
    - Current Sub-task: Detailed analysis of profiling data and implementing optimizations.
    - Plan:
        1. Introduce `pytest-profiling` and gather execution time data. (Completed)
        2. Analyze profiling results to identify slow tests/bottlenecks. (Completed initial overview, now detailed analysis)
        3. Optimize identified test cases (improve mocks, reduce unnecessary computation, review fixtures). (In Progress)
        4. Evaluate and implement test parallelization (`pytest-xdist`) if necessary.
        5. Review and optimize fixture scopes.

## Resolved Issues
- `tests/backend/test_main.py::test_chat_query_uses_embedding_strategy_from_metadata` was failing due to mocking issues for `_read_datasources_meta`. Resolved by mocking `open` and `json.load` instead, ensuring the function's internal logic correctly processes controlled data.
- Various `caplog` assertion errors in manager tests: Resolved by checking `record.message` and `record.levelname`.
- `NameError` and `TypeError` in `test_retriever_manager.py` and `test_chunking_manager.py`: Resolved by adding missing imports and correcting kwarg propagation.
- Indentation errors across multiple test files: Resolved.

- (No issues resolved yet)
