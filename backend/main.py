from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
from uuid import uuid4
from contextlib import asynccontextmanager

from . import crud, models, schemas, security
from .database import SessionLocal, engine
from core.ingestion_service import ingestion_service
from core.rag_chain_service import rag_chain_service
from core.vector_store_manager import vector_store_manager
from core.ingestion_service import EmbeddingServiceError

# Create all database tables on startup
models.Base.metadata.create_all(bind=engine)

# --- Lifespan Management for Resources ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, initialize the ChromaDB client
    vector_store_manager.initialize_client()
    yield
    # On shutdown, you could add cleanup code here if needed

# --- FastAPI App Instance ---
app = FastAPI(lifespan=lifespan)

# --- Dependencies ---

def get_db():
    """Dependency to get a DB session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Dependency to get the current user from a token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = security.decode_access_token(token, credentials_exception)
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    user = crud.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

# --- API Endpoints ---

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.post("/users/me/datasets/", response_model=schemas.Dataset, status_code=status.HTTP_201_CREATED)
def create_dataset_for_user(dataset: schemas.DatasetCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return crud.create_dataset(db=db, dataset=dataset, owner_id=current_user.id)

@app.get("/users/me/datasets/", response_model=List[schemas.Dataset])
def read_datasets_for_user(skip: int = 0, limit: int = 100, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return crud.get_datasets_by_user(db, owner_id=current_user.id, skip=skip, limit=limit)

@app.delete("/users/me/datasets/{dataset_id}", response_model=schemas.Dataset)
def delete_dataset_for_user(dataset_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_dataset = crud.delete_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset

@app.post("/users/me/datasets/{dataset_id}/documents/upload/", response_model=schemas.DataSource)
def upload_document_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    unique_filename = f"{uuid4()}_{file.filename}"
    file_path = os.path.join(upload_dir, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        data_source_schema = schemas.DataSourceCreate(
            original_filename=file.filename,
            file_type=file.content_type
        )
        db_data_source = crud.create_data_source(
            db=db, 
            data_source=data_source_schema, 
            dataset_id=dataset_id, 
            owner_id=current_user.id, 
            file_path=file_path
        )
        ingestion_service.ingest_file(
            file_path=file_path,
            file_type=file.content_type,
            data_source_id=db_data_source.id,
            dataset_id=dataset_id,
            user_id=current_user.id
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return db_data_source

@app.post("/users/me/datasets/{dataset_id}/documents/upload_batch/", response_model=List[schemas.DataSource])
def upload_documents_to_dataset(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    data_sources = []
    for file in files:
        unique_filename = f"{uuid4()}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            data_source_schema = schemas.DataSourceCreate(
                original_filename=file.filename,
                file_type=file.content_type
            )
            db_data_source = crud.create_data_source(
                db=db, 
                data_source=data_source_schema, 
                dataset_id=dataset_id, 
                owner_id=current_user.id, 
                file_path=file_path
            )
            try:
                ingestion_service.ingest_file(
                    file_path=file_path,
                    file_type=file.content_type,
                    data_source_id=db_data_source.id,
                    dataset_id=dataset_id,
                    user_id=current_user.id
                )
            except EmbeddingServiceError as ese:
                # Return a 503 Service Unavailable with a clear message
                raise HTTPException(status_code=503, detail=str(ese))
            data_sources.append(db_data_source)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return data_sources

@app.post("/users/me/datasets/{dataset_id}/documents/upload_for_parent_document/", response_model=List[schemas.DataSource])
def upload_documents_for_parent_document_retriever(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    data_sources = []
    for file in files:
        unique_filename = f"{uuid4()}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            data_source_schema = schemas.DataSourceCreate(
                original_filename=file.filename,
                file_type=file.content_type
            )
            db_data_source = crud.create_data_source(
                db=db, 
                data_source=data_source_schema, 
                dataset_id=dataset_id, 
                owner_id=current_user.id, 
                file_path=file_path
            )
            ingestion_service.ingest_file(
                file_path=file_path,
                file_type=file.content_type,
                data_source_id=db_data_source.id,
                dataset_id=dataset_id,
                user_id=current_user.id,
                strategy="parent_document"
            )
            data_sources.append(db_data_source)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return data_sources

@app.get("/users/me/datasets/{dataset_id}/documents/", response_model=List[schemas.DataSource])
def read_documents_for_dataset(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # crud.get_data_sources_by_dataset will internally verify that the user owns the dataset
    return crud.get_data_sources_by_dataset(db, dataset_id=dataset_id, owner_id=current_user.id, skip=skip, limit=limit)

@app.delete("/users/me/datasets/{dataset_id}/documents/{document_id}", response_model=schemas.DataSource)
def delete_document_from_dataset(
    dataset_id: int,
    document_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Verify user owns the dataset
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    # 2. Verify the document exists and belongs to the specified dataset
    doc_to_delete = crud.get_data_source(db, data_source_id=document_id, owner_id=current_user.id)
    if not doc_to_delete or doc_to_delete.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Document not found in this dataset")

    # 3. Delete the document
    deleted_doc = crud.delete_data_source(db, data_source_id=document_id, owner_id=current_user.id)
    if not deleted_doc:
        # This case should ideally not be reached if the above checks pass
        raise HTTPException(status_code=404, detail="Document not found")

    return deleted_doc

@app.get("/chat/strategies/")
def get_available_rag_strategies():
    """Returns a list of available RAG strategies from the retriever manager."""
    from core.retriever_manager import retriever_manager
    return {"strategies": list(retriever_manager.strategies.keys())}

@app.post("/chat/query/", response_model=schemas.QueryResponse)
def query_rag_chain(
    query: schemas.QueryRequest,
    current_user: models.User = Depends(get_current_user)
):
    return rag_chain_service.get_rag_response(query, user_id=current_user.id)