import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Adjust import path to match how FastAPI apps are typically structured
# Assuming 'backend.main' can be imported if 'tests' is sibling to 'backend'
# and tests are run from project root.
# If running pytest from within 'tests' directory, this might need adjustment
# or PYTHONPATH manipulation.
from backend.main import app  # Direct import of the app instance

# Import models and other necessary components
from backend.models import User, Token
from core.rag_chain import AVAILABLE_RAG_STRATEGIES # For testing strategies

client = TestClient(app)

# --- Test User Data ---
test_username = "testuser_main"
test_email = "testuser_main@example.com"
test_password = "testpassword123"

# --- Fixture for a test user and token ---
@pytest.fixture(scope="module")
def test_user_token():
    # Create user
    response = client.post(
        "/users/",
        json={"username": test_username, "email": test_email, "password": test_password},
    )
    # Handle case where user might already exist from previous test runs if DB is not cleared
    if response.status_code == 400 and "already exists" in response.json().get("detail", ""):
        # User already exists, try to log in to get token
        pass
    elif response.status_code != 200:
        pytest.fail(f"Failed to create test user: {response.status_code} - {response.text}")

    # Log in to get token
    login_response = client.post(
        "/token",
        data={"username": test_username, "password": test_password},
    )
    if login_response.status_code != 200:
        pytest.fail(f"Failed to log in test user: {login_response.status_code} - {login_response.text}")

    token_data = login_response.json()
    return {"access_token": token_data["access_token"], "token_type": token_data["token_type"]}

# --- Test Cases for /chat/query/ endpoint ---

# Mock the get_rag_response function from core.rag_chain
# This is important to isolate backend API tests from the RAG logic itself.
@patch("backend.main.get_rag_response")
def test_chat_query_successful_basic_strategy(mock_get_rag_response, test_user_token):
    mock_get_rag_response.return_value = "This is a mock RAG response."

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "What is MochiRAG?",
        "rag_strategy": "basic" # Explicitly testing basic
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["answer"] == "This is a mock RAG response."
    assert response_data["strategy_used"] == "basic"

    # Verify that get_rag_response was called with correct parameters
    mock_get_rag_response.assert_called_once()
    call_args = mock_get_rag_response.call_args[1] # Get keyword arguments
    assert call_args["question"] == query_payload["question"]
    assert call_args["rag_strategy"] == query_payload["rag_strategy"]
    # user_id is also passed, but it's harder to assert its exact value here without more setup
    # We can assume auth.get_current_active_user works if other tests pass

@patch("backend.main.get_rag_response")
def test_chat_query_successful_another_strategy(mock_get_rag_response, test_user_token):
    # Test with another valid strategy, e.g., "multi_query"
    if "multi_query" not in AVAILABLE_RAG_STRATEGIES:
        pytest.skip("multi_query strategy not in AVAILABLE_RAG_STRATEGIES, skipping test.")

    mock_get_rag_response.return_value = "Mock response from multi_query."

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "Tell me more about MochiRAG features.",
        "rag_strategy": "multi_query"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["answer"] == "Mock response from multi_query."
    assert response_data["strategy_used"] == "multi_query"

    mock_get_rag_response.assert_called_once_with(
        user_id=mock_get_rag_response.call_args[1]['user_id'], # Keep the dynamic user_id
        question=query_payload["question"],
        data_source_ids=None, # Default if not provided
        rag_strategy=query_payload["rag_strategy"]
    )

def test_chat_query_invalid_strategy(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "This should fail.",
        "rag_strategy": "non_existent_strategy_123"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 400 # Bad Request due to invalid strategy
    response_data = response.json()
    assert "Invalid RAG strategy" in response_data["detail"]
    assert "non_existent_strategy_123" in response_data["detail"]

def test_chat_query_missing_question(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        # "question": "Missing question here", # Question is intentionally missing
        "rag_strategy": "basic"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation
    # FastAPI/Pydantic should catch this before our custom logic.

def test_chat_query_unauthorized():
    query_payload = {"question": "Test question", "rag_strategy": "basic"}
    response = client.post("/chat/query/", json=query_payload) # No auth header
    assert response.status_code == 401 # Unauthorized

@patch("backend.main.get_rag_response")
def test_chat_query_rag_function_exception(mock_get_rag_response, test_user_token):
    # Simulate an exception occurring within the get_rag_response function
    mock_get_rag_response.side_effect = Exception("Internal RAG error")

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "A question that causes an internal error.",
        "rag_strategy": "basic"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 500 # Internal Server Error
    response_data = response.json()
    assert "An unexpected error occurred while processing your query" in response_data["detail"]
    assert "Internal RAG error" in response_data["detail"] # Check if original error message is part of detail

# --- Placeholder for Document Upload and Listing Tests (if relevant to current changes) ---
# These tests would typically reside here but are not directly affected by RAG strategy changes.
# For completeness, one might ensure they still work.

def test_upload_document_unsupported_type(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    # Create a dummy file with an unsupported extension
    files = {'file': ('unsupported.xyz', b'some content', 'application/octet-stream')}

    response = client.post("/documents/upload/", headers=headers, files=files)

    assert response.status_code == 400
    assert "Unsupported file type: 'xyz'" in response.json()["detail"]

# TODO: Add more tests for document upload (txt, md, pdf) and listing,
# ensuring they are not broken by other changes. These would involve mocking
# core.document_processor.load_and_split_document and core.vector_store.add_documents_to_vector_db.

# --- Example of a more involved test for document processing (if needed) ---
# @patch("backend.main.load_and_split_document")
# @patch("backend.main.add_documents_to_vector_db")
# def test_upload_document_successful_txt(mock_add_to_db, mock_load_split, test_user_token):
#     mock_load_split.return_value = [MagicMock()] # Return a list with one dummy Document chunk
#     mock_add_to_db.return_value = None

#     headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
#     file_content = b"This is a test text file."
#     files = {'file': ('test.txt', file_content, 'text/plain')}

#     response = client.post("/documents/upload/", headers=headers, files=files)

#     assert response.status_code == 200
#     response_data = response.json()
#     assert response_data["original_filename"] == "test.txt"
#     assert response_data["status"] == "processed"
#     assert response_data["chunk_count"] == 1 # Based on mock_load_split

#     mock_load_split.assert_called_once()
#     # Path will be some temp path, so difficult to assert exactly without more mocking of Path/shutil
#     # Can assert that the file_type was 'txt'
#     assert mock_load_split.call_args[0][1] == 'txt'

#     mock_add_to_db.assert_called_once()
#     # Can assert user_id, data_source_id, and documents structure if needed.
#     assert mock_add_to_db.call_args[1]['user_id'] is not None # User ID is dynamic
#     assert mock_add_to_db.call_args[1]['data_source_id'].startswith("test.txt_")


if __name__ == "__main__":
    pytest.main()
