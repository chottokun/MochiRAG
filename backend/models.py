from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid

class UserInDB(BaseModel):
    user_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    username: str
    email: str
    hashed_password: str
    disabled: bool = False

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[uuid.UUID] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    user_id: uuid.UUID
    username: str
    email: str
    disabled: Optional[bool] = None

class DataSourceMeta(BaseModel):
    data_source_id: str # Typically the filename or a unique ID for the source
    original_filename: str
    status: str # e.g., "uploaded", "processing", "processed", "failed"
    uploaded_at: str # ISO format timestamp
    chunk_count: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None

class ChatQueryRequest(BaseModel):
    question: str
    data_source_ids: Optional[List[str]] = None

class ChatQueryResponse(BaseModel):
    answer: str
    # sources: Optional[List[Dict[str, Any]]] = None # Placeholder for future
