import datetime
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Any, Dict

# --- DataSource Schemas ---

class DataSourceBase(BaseModel):
    original_filename: str
    file_type: str

class DataSourceCreate(DataSourceBase):
    pass

class DataSource(DataSourceBase):
    id: int
    dataset_id: int
    upload_date: datetime.datetime

    class Config:
        from_attributes = True

# --- Dataset Schemas ---

class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetCreate(DatasetBase):
    pass

class Dataset(DatasetBase):
    id: int
    owner_id: int
    data_sources: List[DataSource] = []

    class Config:
        from_attributes = True

# --- User Schemas ---

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    datasets: List[Dataset] = []

    class Config:
        from_attributes = True

# --- Token Schemas for Authentication ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# --- Chat Schemas ---

class QueryRequest(BaseModel):
    query: str
    strategy: str = "basic"
    dataset_ids: Optional[List[int]] = None

class Source(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
