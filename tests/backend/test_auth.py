import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app # Import the FastAPI app instance
from backend.models import UserCreate
from backend.auth import get_password_hash, create_user_in_db # For creating test user

client = TestClient(app)

TEST_USER_USERNAME = "testuser"
TEST_USER_EMAIL = "testuser@example.com"
TEST_USER_PASSWORD = "testpassword"

@pytest.fixture(scope="function") # Use "function" scope for cleaner tests
def test_user_db(tmp_path, monkeypatch):
    """
    Fixture to set up a temporary users JSON database for testing.
    - Creates a temporary test_users.json.
    - Monkeypatches backend.auth.USERS_DB_PATH to use this temporary file.
    - Optionally pre-populates it with a test user.
    - Cleans up the temporary file after the test.
    """
    test_db_path = tmp_path / "test_users.json"

    # Ensure the directory for the test_db_path exists (tmp_path itself)
    # Initialize with an empty list or specific test users
    initial_users = []
    with open(test_db_path, "w") as f:
        json.dump(initial_users, f)

    monkeypatch.setattr("backend.auth.USERS_DB_PATH", test_db_path)

    # It's often better to create users within each test if needed,
    # or have specific fixtures for users.
    # For login tests, we need a pre-existing user.
    # Let's create one here for convenience in this fixture.
    try:
        user_create_data = UserCreate(
            username=TEST_USER_USERNAME,
            email=TEST_USER_EMAIL,
            password=TEST_USER_PASSWORD
        )
        # Directly use create_user_in_db which now writes to the monkeypatched path
        created_user = create_user_in_db(user_create_data)
        # print(f"Test user '{created_user.username}' created in {test_db_path}")
    except ValueError as e:
        # This might happen if tests run in parallel and try to create the same user,
        # or if a previous test run didn't clean up properly (though tmp_path helps).
        # For now, we'll assume it's okay if user already exists from a prior step in this fixture.
        # print(f"Warning: Could not create test user, possibly already exists: {e}")
        pass


    yield test_db_path # Provide the path if needed, though usually not directly used by tests

    # Teardown: tmp_path handles cleanup of the directory and its contents

# --- Tests for /token endpoint ---

def test_login_for_access_token(test_user_db):
    """Test successful login and token generation."""
    response = client.post(
        "/token",
        data={"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_with_wrong_password(test_user_db):
    """Test login with incorrect password."""
    response = client.post(
        "/token",
        data={"username": TEST_USER_USERNAME, "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


def test_login_with_nonexistent_user(test_user_db):
    """Test login with a username that does not exist."""
    response = client.post(
        "/token",
        data={"username": "nonexistentuser", "password": "anypassword"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


# --- Tests for /users/me endpoint ---

def test_read_users_me_success(test_user_db):
    """Test successful retrieval of current user details."""
    # 1. Login to get a token
    login_response = client.post(
        "/token",
        data={"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD}
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    # 2. Request /users/me with the token
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/users/me", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == TEST_USER_USERNAME
    assert data["email"] == TEST_USER_EMAIL
    assert "hashed_password" not in data # Ensure sensitive info is not returned

def test_read_users_me_no_token(test_user_db):
    """Test /users/me endpoint without providing an access token."""
    response = client.get("/users/me")
    assert response.status_code == 401 # Expect 401 Unauthorized
    # FastAPI's default for missing OAuth2 token is often 403, but Depends(oauth2_scheme) should yield 401
    # Let's check the detail if possible, though it might vary.
    # Example: assert response.json()["detail"] == "Not authenticated"
    # For now, status code 401 is the primary check.

def test_read_users_me_invalid_token(test_user_db):
    """Test /users/me endpoint with an invalid or malformed token."""
    headers = {"Authorization": "Bearer invalidtoken123"}
    response = client.get("/users/me", headers=headers)
    assert response.status_code == 401 # Expect 401 Unauthorized
    # Detail might include "Could not validate credentials" or similar
    # assert response.json()["detail"] == "Could not validate credentials"

# --- Test for user creation endpoint (as a prerequisite for robust auth tests) ---
# This is more of a test for /users/ endpoint but good to have here for completeness of auth flow.
def test_create_new_user_for_auth_tests(tmp_path, monkeypatch):
    """Test creating a new user, ensuring it doesn't conflict with the fixture user."""
    test_db_path = tmp_path / "another_test_users.json"
    with open(test_db_path, "w") as f:
        json.dump([], f) # Start with empty DB for this specific test
    monkeypatch.setattr("backend.auth.USERS_DB_PATH", test_db_path)

    new_username = "newtestuser"
    new_email = "new@example.com"
    new_password = "newpassword"

    response = client.post(
        "/users/",
        json={"username": new_username, "email": new_email, "password": new_password}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == new_username
    assert data["email"] == new_email
    assert "id" not in data # Assuming User model for response doesn't include raw id
    assert "user_id" in data # Should have user_id (UUID)

    # Verify user can login (optional, but good check)
    login_resp = client.post("/token", data={"username": new_username, "password": new_password})
    assert login_resp.status_code == 200
    assert "access_token" in login_resp.json()

    # Verify user already exists if tried again
    response_again = client.post(
        "/users/",
        json={"username": new_username, "email": new_email, "password": new_password}
    )
    assert response_again.status_code == 400 # Bad Request
    assert "already exists" in response_again.json()["detail"]

    # Clean up by removing the specific test DB file if not handled by tmp_path (it is)
    # if test_db_path.exists():
    #     test_db_path.unlink()
