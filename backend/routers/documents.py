from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from uuid import uuid4

from .. import crud, schemas, models
from ..dependencies import get_db, get_current_user
from core.ingestion_service import ingestion_service, EmbeddingServiceError

router = APIRouter(prefix="/users/me/datasets/{dataset_id}/documents", tags=["documents"])

UPLOAD_DIR = "temp_uploads"

def save_upload_file(upload_file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    unique_filename = f"{uuid4()}_{upload_file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@router.post("/upload/", response_model=schemas.DataSource)
def upload_document_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    file_path = save_upload_file(file)
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
        return db_data_source
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/upload_batch/", response_model=List[schemas.DataSource])
def upload_documents_to_dataset(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    data_sources = []
    for file in files:
        file_path = save_upload_file(file)
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
                raise HTTPException(status_code=503, detail=str(ese))
            data_sources.append(db_data_source)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return data_sources

@router.post("/upload_for_parent_document/", response_model=List[schemas.DataSource])
def upload_documents_for_parent_document_retriever(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    data_sources = []
    for file in files:
        file_path = save_upload_file(file)
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

@router.get("/", response_model=List[schemas.DataSource])
def read_documents_for_dataset(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_data_sources_by_dataset(db, dataset_id=dataset_id, owner_id=current_user.id, skip=skip, limit=limit)

@router.delete("/{document_id}", response_model=schemas.DataSource)
def delete_document_from_dataset(
    dataset_id: int,
    document_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    doc_to_delete = crud.get_data_source(db, data_source_id=document_id, owner_id=current_user.id)
    if not doc_to_delete or doc_to_delete.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Document not found in this dataset")

    deleted_doc = crud.delete_data_source(db, data_source_id=document_id, owner_id=current_user.id)
    if not deleted_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return deleted_doc
