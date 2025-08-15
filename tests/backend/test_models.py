import pytest
from sqlalchemy import inspect, Integer, String
from sqlalchemy.orm import Mapped

def test_user_model_exists():
    """
    Tests if the User model can be imported and has the correct columns.
    """
    try:
        from backend.models import User
    except ImportError:
        pytest.fail("Could not import User model from backend.models")

    inspector = inspect(User)
    columns = [c.name for c in inspector.columns]

    assert "id" in columns
    assert "email" in columns
    assert "hashed_password" in columns

    # Check types, especially for mapped columns
    id_col = inspector.columns['id']
    assert isinstance(id_col.type, Integer) # This will fail until sqlalchemy is imported

    email_col = inspector.columns['email']
    assert isinstance(email_col.type, String) # This will also fail

    # Check relationships if any in the future
    # assert 'datasets' in inspector.relationships
