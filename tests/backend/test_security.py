import pytest
from datetime import timedelta
from jose import jwt

# We will import these from backend.security once they are created
# For now, this import will fail, which is the point of TDD
from backend.security import create_access_token, ALGORITHM, SECRET_KEY, verify_password

def test_password_hashing():
    """
    Tests the password hashing and verification functions.
    """
    from backend.security import get_password_hash

    password = "mysecretpassword"
    hashed_password = get_password_hash(password)

    assert hashed_password != password
    assert verify_password(password, hashed_password)
    assert not verify_password("wrongpassword", hashed_password)

def test_create_access_token():
    """
    Tests the creation of a JWT access token.
    """
    data = {"sub": "testuser@example.com"}
    expires_delta = timedelta(minutes=15)

    token = create_access_token(data, expires_delta)
    assert isinstance(token, str)

    # Decode the token to verify its contents
    decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    assert decoded_token["sub"] == "testuser@example.com"
    assert "exp" in decoded_token

    # We can also test that a token expires correctly, but that's more complex.
    # For now, just checking existence is enough.
