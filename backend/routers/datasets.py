from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import json
from .. import crud, schemas, models
from ..dependencies import get_db, get_current_user

router = APIRouter(prefix="/users/me/datasets", tags=["datasets"])

@router.post("/", response_model=schemas.Dataset, status_code=status.HTTP_201_CREATED)
def create_dataset_for_user(
    dataset: schemas.DatasetCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return crud.create_dataset(db=db, dataset=dataset, owner_id=current_user.id)

@router.get("/", response_model=List[schemas.Dataset])
def read_datasets_for_user(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Get personal datasets from the database
    personal_datasets = crud.get_datasets_by_user(db, owner_id=current_user.id, skip=skip, limit=limit)

    # 2. Load shared datasets from the JSON file
    shared_datasets = []
    try:
        with open("shared_dbs.json", "r") as f:
            shared_dbs_data = json.load(f)
            for item in shared_dbs_data:
                shared_datasets.append(
                    schemas.Dataset(
                        id=item["id"],
                        name=item["name"],
                        description=item.get("description"),
                        owner_id=-1,
                        data_sources=[]
                    )
                )
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return personal_datasets + shared_datasets

@router.delete("/{dataset_id}", response_model=schemas.Dataset)
def delete_dataset_for_user(
    dataset_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_dataset = crud.delete_dataset(db, dataset_id=dataset_id, owner_id=current_user.id)
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset
