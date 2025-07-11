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
    embedding_strategy_used: Optional[str] = None # 追加: 使用されたエンベディング戦略
    chunking_strategy_used: Optional[str] = None # 追加: 使用されたチャンキング戦略
    chunking_config_used: Optional[Dict[str, Any]] = None # 追加: 使用されたチャンキング設定

class DocumentUploadRequest(BaseModel): # 新規: ドキュメントアップロード時のリクエストボディ
    embedding_strategy: Optional[str] = None # 指定がなければデフォルトを使用
    chunking_strategy: Optional[str] = None  # 指定がなければデフォルトを使用
    chunking_params: Optional[Dict[str, Any]] = None # チャンキング戦略のパラメータ (例: chunk_size)


class ChatQueryRequest(BaseModel):
    question: str
    data_source_ids: Optional[List[str]] = None
    rag_strategy: Optional[str] = "basic" # Default to basic strategy

class ChatQueryResponse(BaseModel):
    answer: str
    strategy_used: Optional[str] = None # To confirm which strategy was applied
    sources: Optional[List[Dict[str, Any]]] = None # List of source documents, each a dict with "page_content" and "metadata"
