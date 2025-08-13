import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import crud, models, schemas
from backend.database import Base

# --- Test Database Setup ---
# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pytest fixture to set up and tear down the database for each test function
@pytest.fixture()
def db():
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)

# --- Tests ---

def test_create_and_get_user(db):
    user_in = schemas.UserCreate(email="test@example.com", password="password123")
    db_user = crud.create_user(db, user_in)
    
    assert db_user.email == user_in.email
    assert hasattr(db_user, "hashed_password")

    get_db_user = crud.get_user_by_email(db, email=user_in.email)
    assert get_db_user.id == db_user.id


def test_create_and_get_dataset(db):
    # 1. Create a user first
    user_in = schemas.UserCreate(email="owner@example.com", password="password123")
    db_user = crud.create_user(db, user_in)

    # 2. Create a dataset for that user
    dataset_in = schemas.DatasetCreate(name="Test Dataset", description="A test dataset")
    db_dataset = crud.create_dataset(db, dataset_in, owner_id=db_user.id)

    assert db_dataset.name == dataset_in.name
    assert db_dataset.owner_id == db_user.id

    # 3. Verify that the user can retrieve their dataset
    user_datasets = crud.get_datasets_by_user(db, owner_id=db_user.id)
    assert len(user_datasets) == 1
    assert user_datasets[0].id == db_dataset.id

def test_multi_tenancy_isolation(db):
    # 1. Create two users
    user1 = crud.create_user(db, schemas.UserCreate(email="user1@example.com", password="pw1"))
    user2 = crud.create_user(db, schemas.UserCreate(email="user2@example.com", password="pw2"))

    # 2. Create a dataset for user1
    dataset1_in = schemas.DatasetCreate(name="Dataset 1")
    dataset1 = crud.create_dataset(db, dataset1_in, owner_id=user1.id)

    # 3. Assert that user2 CANNOT retrieve user1's dataset
    user2_datasets = crud.get_datasets_by_user(db, owner_id=user2.id)
    assert len(user2_datasets) == 0

    # 4. Assert that user2 CANNOT get user1's dataset by its ID
    retrieved_dataset = crud.get_dataset(db, dataset_id=dataset1.id, owner_id=user2.id)
    assert retrieved_dataset is None

    # 5. Assert that user1 CAN get their dataset by ID
    retrieved_dataset = crud.get_dataset(db, dataset_id=dataset1.id, owner_id=user1.id)
    assert retrieved_dataset is not None
    assert retrieved_dataset.id == dataset1.id

def test_delete_dataset(db):
    # 1. Create user and dataset
    user = crud.create_user(db, schemas.UserCreate(email="user@test.com", password="pw"))
    dataset = crud.create_dataset(db, schemas.DatasetCreate(name="To Be Deleted"), owner_id=user.id)
    dataset_id = dataset.id

    # 2. Delete the dataset
    deleted = crud.delete_dataset(db, dataset_id=dataset_id, owner_id=user.id)
    assert deleted.id == dataset_id

    # 3. Verify it's gone
    assert crud.get_dataset(db, dataset_id=dataset_id, owner_id=user.id) is None