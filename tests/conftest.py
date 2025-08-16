import pytest
import shutil
import tempfile
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator

from backend.main import app, get_db
from backend.database import Base
from core.vector_store_manager import vector_store_manager

# --- Test Database Fixture ---

@pytest.fixture(scope="session")
def test_db_session() -> Generator[sessionmaker, None, None]:
    """
    Pytest fixture to create a temporary SQLite database for a test session.
    Yields a sessionmaker instance for creating new sessions.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = f"{temp_dir}/test.db"
        SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_path}"

        engine = create_engine(
            SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
        )
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        # Create tables in the test database
        Base.metadata.create_all(bind=engine)

        yield TestingSessionLocal

        # Teardown: tables are dropped implicitly with the temp directory

# --- ChromaDB Vector Store Fixture ---

@pytest.fixture(scope="session")
def test_vector_store() -> Generator[None, None, None]:
    """
    Pytest fixture to initialize ChromaDB in a temporary directory for a test session.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_manager.initialize_client(db_path=temp_dir)
        yield
        # Teardown: ChromaDB files are removed with the temp directory

# --- Override get_db Dependency Fixture ---

@pytest.fixture(autouse=True)
def override_get_db(test_db_session: sessionmaker) -> Generator[None, None, None]:
    """
    Pytest fixture to override the `get_db` dependency in the FastAPI app
    with a session from the temporary test database.
    """
    def _override_get_db():
        try:
            db = test_db_session()
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield
    # Teardown: remove the override after the test
    del app.dependency_overrides[get_db]

# --- Test Client Fixture ---

@pytest.fixture
async def test_client(test_vector_store, override_get_db) -> AsyncClient:
    """
    Pytest fixture to provide a test client for the FastAPI application.
    This client is configured to use the temporary database and vector store.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
