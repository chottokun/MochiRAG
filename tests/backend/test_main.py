import pytest
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
