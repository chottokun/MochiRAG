import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    data_sources = relationship("DataSource", back_populates="owner", cascade="all, delete-orphan")
    evolved_contexts = relationship("EvolvedContext", back_populates="owner", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=True)
    
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="datasets")
    
    data_sources = relationship("DataSource", back_populates="dataset", cascade="all, delete-orphan")

class DataSource(Base):
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False, unique=True)
    file_type = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="data_sources")

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    dataset = relationship("Dataset", back_populates="data_sources")

    parent_documents = relationship("ParentDocument", back_populates="data_source", cascade="all, delete-orphan")

class ParentDocument(Base):
    __tablename__ = "parent_documents"

    id = Column(String, primary_key=True, index=True)
    content = Column(String, nullable=False)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    
    data_source = relationship("DataSource", back_populates="parent_documents")

class EvolvedContext(Base):
    __tablename__ = "evolved_contexts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    topic = Column(String, index=True, nullable=True) # Can be used for targeted retrieval
    effectiveness_score = Column(Integer, default=0, nullable=False) # Simple scoring
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="evolved_contexts")
