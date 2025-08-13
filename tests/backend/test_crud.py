import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

# Will be created in the next step
from backend import crud, models
from backend.schemas import UserCreate # This will also need to be created

@pytest.fixture
def mock_db_session():
    """Provides a mock SQLAlchemy session."""
    db = MagicMock(spec=Session)
    # Configure the mock query chain
    db.query.return_value.filter.return_value.first.return_value = None
    return db

def test_get_user_by_email(mock_db_session: Session):
    """
    Tests retrieving a user by email.
    """
    # --- Arrange ---
    # Configure the mock to return a user when a specific email is queried
    test_email = "test@example.com"
    expected_user = models.User(id=1, email=test_email, hashed_password="fake_hash")
    mock_db_session.query.return_value.filter.return_value.first.return_value = expected_user

    # --- Act ---
    user = crud.get_user_by_email(db=mock_db_session, email=test_email)

    # --- Assert ---
    assert user is not None
    assert user.email == test_email
    mock_db_session.query.return_value.filter.assert_called_once()


def test_create_user(mock_db_session: Session):
    """
    Tests creating a new user.
    """
    # --- Arrange ---
    user_data = UserCreate(email="newuser@example.com", password="newpassword123")

    # --- Act ---
    created_user = crud.create_user(db=mock_db_session, user=user_data)

    # --- Assert ---
    assert created_user is not None
    assert created_user.email == user_data.email
    # The password should be hashed, not plain text
    assert created_user.hashed_password != user_data.password

    # Check that the user was added to the session and the session was committed
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()
