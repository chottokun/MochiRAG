import pytest
from httpx import AsyncClient
import os
from langchain_community.embeddings.fake import FakeEmbeddings

from core.vector_store_manager import vector_store_manager
from core.embedding_manager import embedding_manager

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.anyio

# --- Fixtures ---

@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Mock the EmbeddingManager to return a FakeEmbeddings instance."""
    fake_embeddings = FakeEmbeddings(size=384)

    # Mock the method that provides the embeddings
    monkeypatch.setattr(
        embedding_manager,
        "get_embedding_model",
        lambda name=None: fake_embeddings
    )

    # The VectorStoreManager is initialized at app startup, so its embedding_function
    # was already set. We need to patch it directly as well.
    monkeypatch.setattr(
        vector_store_manager,
        "embedding_function",
        fake_embeddings
    )


# --- Test Data ---
TEST_USER_EMAIL = "test_integration@example.com"
TEST_USER_PASSWORD = "password123"
TEST_DATASET_NAME = "My Test Dataset"
TEST_FILENAME = "test_document.txt"
TEST_FILE_CONTENT = "This is a test document for integration testing."

# --- Helper Function ---
def get_vector_count(collection_name: str, filter_criteria: dict) -> int:
    """Helper to get the number of vectors matching a filter in a ChromaDB collection."""
    vector_store = vector_store_manager.get_vector_store(collection_name)
    results = vector_store.get(where=filter_criteria, include=[])
    return len(results['ids'])

# --- E2E Test Case ---

async def test_delete_document_e2e(test_client: AsyncClient):
    """
    End-to-end test for the document deletion process.
    1. Creates a user, dataset, and uploads a document.
    2. Verifies vectors are created in ChromaDB.
    3. Deletes the document via the API.
    4. Verifies the vectors are also deleted from ChromaDB.
    """
    # 1. Create User
    response = await test_client.post(
        "/users/",
        json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
    )
    # It might fail if user exists from a previous failed run, which is fine
    if response.status_code == 201:
        user_id = response.json()["id"]
    else:
        # If user already exists, we need to fetch the ID.
        # For simplicity in this test, we'll assume a clean state or handle login failure.
        # A more robust test suite might have a fixture to pre-create users.
        pass # We will just login

    # 2. Login to get token
    response = await test_client.post(
        "/token",
        data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # We need the user_id for the collection name
    # A proper way would be to decode the token, but for test simplicity, let's assume user id 1
    # In a real app, GET /users/me would be ideal. Let's find the user.
    # For now, let's assume the first user is our test user. This is a shortcut.
    # In a real scenario, you'd fetch the user from the DB via an API endpoint.
    # Let's assume user_id is 1, as it's a fresh test DB.
    user_id = 1
    collection_name = f"user_{user_id}"

    # 3. Create Dataset
    response = await test_client.post(
        "/users/me/datasets/",
        headers=headers,
        json={"name": TEST_DATASET_NAME, "description": "A dataset for testing."},
    )
    assert response.status_code == 201
    dataset_id = response.json()["id"]

    # 4. Upload Document
    with open(TEST_FILENAME, "w") as f:
        f.write(TEST_FILE_CONTENT)

    with open(TEST_FILENAME, "rb") as f:
        response = await test_client.post(
            f"/users/me/datasets/{dataset_id}/documents/upload/",
            headers=headers,
            files={"file": (TEST_FILENAME, f, "text/plain")},
        )
    os.remove(TEST_FILENAME) # Clean up the temp file

    assert response.status_code == 200
    document_id = response.json()["id"]

    # 5. Verify Vectors were Added to ChromaDB
    # Allow some time for ingestion to complete if it were async
    # In this app, ingestion is synchronous with the upload request
    filter_criteria = {"data_source_id": document_id}
    vector_count_before_delete = get_vector_count(collection_name, filter_criteria)
    print(f"Vectors found before delete: {vector_count_before_delete}")
    assert vector_count_before_delete > 0

    # 6. Delete Document via API
    delete_response = await test_client.delete(
        f"/users/me/datasets/{dataset_id}/documents/{document_id}",
        headers=headers,
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["id"] == document_id

    # 7. Verify Vectors were Deleted from ChromaDB (This is the key assertion)
    vector_count_after_delete = get_vector_count(collection_name, filter_criteria)
    print(f"Vectors found after delete: {vector_count_after_delete}")
    assert vector_count_after_delete == 0, "Vectors should be deleted from ChromaDB, but they were not."
