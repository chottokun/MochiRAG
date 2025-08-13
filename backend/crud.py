from sqlalchemy.orm import Session

from . import models, schemas, security

# --- User CRUD ---

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Dataset CRUD ---

def get_dataset(db: Session, dataset_id: int, owner_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id, models.Dataset.owner_id == owner_id).first()

def get_datasets_by_user(db: Session, owner_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Dataset).filter(models.Dataset.owner_id == owner_id).offset(skip).limit(limit).all()

def create_dataset(db: Session, dataset: schemas.DatasetCreate, owner_id: int):
    db_dataset = models.Dataset(**dataset.model_dump(), owner_id=owner_id)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def delete_dataset(db: Session, dataset_id: int, owner_id: int):
    db_dataset = get_dataset(db, dataset_id, owner_id)
    if db_dataset:
        db.delete(db_dataset)
        db.commit()
    return db_dataset

# --- DataSource CRUD ---

def get_data_source(db: Session, data_source_id: int, owner_id: int):
    return db.query(models.DataSource).filter(models.DataSource.id == data_source_id, models.DataSource.owner_id == owner_id).first()

def get_data_sources_by_dataset(db: Session, dataset_id: int, owner_id: int, skip: int = 0, limit: int = 100):
    # Ensure the user owns the dataset they are querying
    if not get_dataset(db, dataset_id, owner_id):
        return []
    return db.query(models.DataSource).filter(models.DataSource.dataset_id == dataset_id).offset(skip).limit(limit).all()

def create_data_source(db: Session, data_source: schemas.DataSourceCreate, dataset_id: int, owner_id: int, file_path: str):
    # Check if a data source with the same file_path already exists
    db_data_source = get_data_source_by_file_path(db, file_path=file_path)
    if db_data_source:
        # If it exists, return the existing one instead of creating a duplicate
        return db_data_source

    db_data_source = models.DataSource(
        **data_source.model_dump(), 
        dataset_id=dataset_id, 
        owner_id=owner_id, 
        file_path=file_path
    )
    db.add(db_data_source)
    db.commit()
    db.refresh(db_data_source)
    return db_data_source

def delete_data_source(db: Session, data_source_id: int, owner_id: int):
    db_data_source = get_data_source(db, data_source_id, owner_id)
    if db_data_source:
        db.delete(db_data_source)
        db.commit()
    return db_data_source

def get_data_source_by_file_path(db: Session, file_path: str):
    return db.query(models.DataSource).filter(models.DataSource.file_path == file_path).first()