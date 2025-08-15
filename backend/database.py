from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# For simplicity, we'll use a local SQLite database.
# In a production environment, this would be a more robust database like PostgreSQL.
SQLALCHEMY_DATABASE_URL = "sqlite:///./mochirag.db"

engine = create_engine(
    # check_same_thread is only needed for SQLite.
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Each instance of the SessionLocal class will be a database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This Base will be used by our models to inherit from.
Base = declarative_base()
