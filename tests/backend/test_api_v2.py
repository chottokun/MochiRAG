import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app
from backend.database import Base, engine, SessionLocal
from backend import models

# Setup an in-memory database for testing
# (Normally we would use a separate test DB, but for this simulation we override get_db)

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

import uuid

@pytest.mark.anyio
async def test_full_workflow(client):
    random_id = str(uuid.uuid4())[:8]
    email = f"user_{random_id}@example.com"
    # 1. Register a new user
    user_data = {"email": email, "password": "securepassword"}
    response = await client.post("/users/", json=user_data)
    assert response.status_code == status.HTTP_201_CREATED

    # 2. Login to get token
    login_data = {"username": email, "password": "securepassword"}
    response = await client.post("/token", data=login_data)
    assert response.status_code == status.HTTP_200_OK
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Create a dataset
    dataset_data = {"name": "Test Dataset", "description": "A test dataset"}
    response = await client.post("/users/me/datasets/", json=dataset_data, headers=headers)
    assert response.status_code == status.HTTP_201_CREATED
    dataset_id = response.json()["id"]

    # 4. List datasets (should include shared ones)
    response = await client.get("/users/me/datasets/", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    datasets = response.json()
    assert any(ds["name"] == "Test Dataset" for ds in datasets)

    # 5. Upload a document (Mock Ingestion Service)
    with patch("backend.routers.documents.ingestion_service.ingest_file") as mock_ingest:
        files = {"file": ("test.txt", b"hello world", "text/plain")}
        response = await client.post(
            f"/users/me/datasets/{dataset_id}/documents/upload/",
            files=files,
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["original_filename"] == "test.txt"
        mock_ingest.assert_called_once()

    # 6. Chat Query (Mock RAG Chain Service)
    with patch("backend.routers.chat.rag_chain_service.get_rag_response") as mock_rag:
        mock_rag.return_value = MagicMock(
            answer="This is a mocked answer.",
            sources=[],
            topic=None
        )
        query_data = {"query": "What is in the test file?", "dataset_ids": [dataset_id]}
        response = await client.post("/chat/query/", json=query_data, headers=headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["answer"] == "This is a mocked answer."

    # 7. Delete dataset
    response = await client.delete(f"/users/me/datasets/{dataset_id}", headers=headers)
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.anyio
async def test_unauthorized_access(client):
    response = await client.get("/users/me/datasets/")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_global_exception_handler():
    # We use the synchronous TestClient to easily disable raising exceptions
    client = TestClient(app, raise_server_exceptions=False)
    random_id = str(uuid.uuid4())[:8]
    email = f"error_{random_id}@example.com"
    # We trigger an unhandled exception by patching the CRUD function used by the router
    with patch("backend.crud.create_user", side_effect=Exception("Unexpected Error")):
        user_data = {"email": email, "password": "password"}
        response = client.post("/users/", json=user_data)
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal Server Error"}
