import pytest
import os
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from fastapi import Depends
from sqlalchemy.orm import Session

from backend.main import app, get_db
from backend import schemas, crud, security

# Mock the get_db dependency
def override_get_db():
    db = MagicMock(spec=Session)
    yield db

app.dependency_overrides[get_db] = override_get_db


@pytest.mark.anyio
async def test_create_user(monkeypatch):
    # Mock the crud functions
    mock_created_user = schemas.User(id=1, email="test@example.com", is_active=True)
    monkeypatch.setattr(crud, "create_user", MagicMock(return_value=mock_created_user))
    monkeypatch.setattr(crud, "get_user_by_email", MagicMock(return_value=None))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/users/",
            json={"email": "test@example.com", "password": "password123"},
        )

    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
    crud.create_user.assert_called_once()


@pytest.mark.anyio
async def test_create_existing_user(monkeypatch):
    monkeypatch.setattr(crud, "get_user_by_email", MagicMock(return_value=True))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/users/",
            json={"email": "test@example.com", "password": "password123"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Email already registered"}


@pytest.mark.anyio
async def test_login_for_access_token(monkeypatch):
    # Mock user and password verification
    mock_user = MagicMock()
    mock_user.email = "test@example.com"
    mock_user.hashed_password = "hashed_password"
    monkeypatch.setattr(crud, "get_user_by_email", MagicMock(return_value=mock_user))
    monkeypatch.setattr(security, "verify_password", MagicMock(return_value=True))

    # Mock token creation
    mock_create_token = MagicMock(return_value="fake_token")
    monkeypatch.setattr(security, "create_access_token", mock_create_token)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/token",
            data={"username": "test@example.com", "password": "password123"}
        )

    assert response.status_code == 200
    assert response.json() == {"access_token": "fake_token", "token_type": "bearer"}
    # Verify that create_access_token was called with the correct subject
    mock_create_token.assert_called_once_with(data={"sub": "test@example.com"})


@pytest.mark.anyio
async def test_read_datasets_for_user_includes_shared_dbs(monkeypatch):
    """
    Test that the /users/me/datasets/ endpoint returns both the user's own
    datasets and the shared datasets defined in shared_dbs.json.
    """
    from backend.main import get_current_user
    from backend import models
    import json

    try:
        # 1. Mock the authenticated user
        mock_user = models.User(id=1, email="test@example.com", hashed_password="fake_password")
        def override_get_current_user():
            return mock_user
        app.dependency_overrides[get_current_user] = override_get_current_user

        # 2. Mock the personal datasets returned from CRUD
        mock_personal_datasets = [
            schemas.Dataset(id=10, name="Personal DB 1", description="My first db", owner_id=1),
            schemas.Dataset(id=11, name="Personal DB 2", description="My second db", owner_id=1),
        ]
        monkeypatch.setattr(crud, "get_datasets_by_user", MagicMock(return_value=mock_personal_datasets))

        # 3. Create the specific shared_dbs.json for this test to ensure isolation
        shared_dbs_data = [{
            "id": -1,
            "name": "Common Test DB",
            "description": "A shared database for testing purposes available to all users.",
            "collection_name": "shared_common_test_db"
        }]
        with open("shared_dbs.json", "w") as f:
            json.dump(shared_dbs_data, f)

        # 4. Make the API call
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # The header is not strictly needed because we override the dependency,
            # but it's good practice for simulating a real call.
            response = await client.get(
                "/users/me/datasets/",
                headers={"Authorization": "Bearer fake-token"}
            )

        # 5. Assert the results
        assert response.status_code == 200
        response_data = response.json()

        # The implementation doesn't exist yet, so this will fail.
        # Expected length = personal (2) + shared (1) = 3
        assert len(response_data) == len(mock_personal_datasets) + len(shared_dbs_data)

        # Check that personal datasets are present
        personal_db_names = {db["name"] for db in response_data if db["id"] > 0}
        assert "Personal DB 1" in personal_db_names
        assert "Personal DB 2" in personal_db_names

        # Check that shared dataset is present
        shared_db_names = {db["name"] for db in response_data if db["id"] < 0}
        assert "Common Test DB" in shared_db_names

    # Clean up the dependency override and the test file
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        if os.path.exists("shared_dbs.json"):
            os.remove("shared_dbs.json")
