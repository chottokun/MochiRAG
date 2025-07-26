import json
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm

from . import auth
from .models import (
    User, UserCreate, Token, DataSourceMeta,
    ChatQueryRequest, ChatQueryResponse,
    Dataset, DatasetCreate, DatasetUpdate
)

try:
    from core.document_processor import SUPPORTED_FILE_TYPES, load_and_split_document
    from core.vector_store_manager import vector_store_manager
    from core.embedding_manager import embedding_manager
    from core.chunking_manager import chunking_manager
    from core.rag_strategies.factory import rag_strategy_factory, get_available_rag_strategies
    from langchain_core.documents import Document
except ImportError as e:
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from core.document_processor import SUPPORTED_FILE_TYPES, load_and_split_document
        from core.vector_store_manager import vector_store_manager
        from core.embedding_manager import embedding_manager
        from core.chunking_manager import chunking_manager
        from core.rag_strategies.factory import rag_strategy_factory, get_available_rag_strategies
        from langchain_core.documents import Document
    except ImportError:
        raise ImportError(f"Could not import core modules. Ensure 'core' is in PYTHONPATH and managers are initialized. Original error: {e}")


app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Metadata Store Configuration ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_META_PATH = DATA_DIR / "datasets_meta.json"
DATASOURCES_META_PATH = DATA_DIR / "datasources_meta.json"
TMP_UPLOAD_DIR = DATA_DIR / "tmp_uploads"
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper functions for Dataset Metadata ---
def _ensure_datasets_meta_file_exists():
    if not DATASETS_META_PATH.exists():
        with open(DATASETS_META_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)

@lru_cache(maxsize=32)
def _read_datasets_meta() -> Dict[str, List[Dataset]]:
    _ensure_datasets_meta_file_exists()
    try:
        with open(DATASETS_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            parsed_data: Dict[str, List[Dataset]] = {}
            for user_id_str, datasets_json_list in data.items():
                parsed_data[user_id_str] = [Dataset(**ds_json) for ds_json in datasets_json_list]
            return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding datasets_meta.json: {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading datasets_meta.json: {e}")
        return {}

def _write_datasets_meta(data: Dict[str, List[Dataset]]):
    _ensure_datasets_meta_file_exists()
    try:
        serializable_data: Dict[str, List[Dict[str, Any]]] = {}
        for user_id_str, datasets_list in data.items():
            serializable_data[user_id_str] = [ds.model_dump(mode="json") for ds in datasets_list]
        with open(DATASETS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
        _read_datasets_meta.cache_clear()
    except Exception as e:
        logger.error(f"Error writing to datasets_meta.json: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save dataset metadata: {str(e)}")

# --- Helper functions for DataSource (File) Metadata ---
def _ensure_datasources_meta_file_exists():
    if not DATASOURCES_META_PATH.exists():
        with open(DATASOURCES_META_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)

@lru_cache(maxsize=32)
def _read_datasources_meta() -> Dict[str, Dict[str, List[DataSourceMeta]]]:
    _ensure_datasources_meta_file_exists()
    try:
        with open(DATASOURCES_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            parsed_data: Dict[str, Dict[str, List[DataSourceMeta]]] = {}
            for user_id_str, user_datasets_json in data.items():
                parsed_data[user_id_str] = {}
                for dataset_id_str, ds_meta_list_json in user_datasets_json.items():
                    parsed_data[user_id_str][dataset_id_str] = [
                        DataSourceMeta(**meta) for meta in ds_meta_list_json
                    ]
            return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding datasources_meta.json: {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading datasources_meta.json: {e}")
        return {}

def _write_datasources_meta(data: Dict[str, Dict[str, List[DataSourceMeta]]]):
    _ensure_datasources_meta_file_exists()
    try:
        serializable_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for user_id_str, user_datasets_pydantic in data.items():
            serializable_data[user_id_str] = {}
            for dataset_id_str, ds_meta_list_pydantic in user_datasets_pydantic.items():
                serializable_data[user_id_str][dataset_id_str] = [
                    meta.model_dump(mode="json") for meta in ds_meta_list_pydantic
                ]
        with open(DATASOURCES_META_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
        _read_datasources_meta.cache_clear()
    except Exception as e:
        logger.error(f"Error writing to datasources_meta.json: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save datasource metadata: {str(e)}")

# Initialize metadata files on startup
_ensure_datasets_meta_file_exists()
_ensure_datasources_meta_file_exists()

# --- Dependency Injection Functions ---

async def get_user_datasets(current_user: User = Depends(auth.get_current_active_user)) -> List[Dataset]:
    return _read_datasets_meta().get(str(current_user.user_id), [])

async def get_user_datasources(current_user: User = Depends(auth.get_current_active_user)) -> Dict[str, Dict[str, List[DataSourceMeta]]]:
    return _read_datasources_meta().get(str(current_user.user_id), {})

async def get_dataset_or_404(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user),
    user_datasets: List[Dataset] = Depends(get_user_datasets)
) -> Dataset:
    dataset = next((ds for ds in user_datasets if ds.dataset_id == dataset_id), None)
    if not dataset:
        logger.warning(f"Dataset {dataset_id} not found for user {current_user.user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset

# --- Endpoints ---

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username, "user_id": str(user.user_id)},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=User)
async def create_user_endpoint(user_data: UserCreate):
    try:
        created_user_in_db = auth.create_user_in_db(user_data)
        return User(**created_user_in_db.model_dump())
    except ValueError as e:
        logger.warning(f"User creation failed: {e}")
        if "already exists" in str(e):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred during user creation.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(auth.get_current_active_user)):
    return current_user

# --- Dataset Endpoints ---

@app.post("/users/me/datasets/", response_model=Dataset, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset_create: DatasetCreate,
    current_user: User = Depends(auth.get_current_active_user),
    user_datasets: List[Dataset] = Depends(get_user_datasets)
):
    user_id_str = str(current_user.user_id)

    if any(ds.name == dataset_create.name for ds in user_datasets):
        logger.warning(f"Attempted to create duplicate dataset name \'{dataset_create.name}\' for user {user_id_str}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset with name \'{dataset_create.name}\' already exists."
        )

    new_dataset = Dataset(
        user_id=current_user.user_id,
        name=dataset_create.name,
        description=dataset_create.description
    )
    all_datasets_meta = _read_datasets_meta()
    all_datasets_meta.setdefault(user_id_str, []).append(new_dataset)
    _write_datasets_meta(all_datasets_meta)
    logger.info(f"Dataset \'{new_dataset.name}\' ({new_dataset.dataset_id}) created for user {user_id_str}")
    return new_dataset

@app.get("/users/me/datasets/", response_model=List[Dataset])
async def list_datasets(user_datasets: List[Dataset] = Depends(get_user_datasets)):
    return user_datasets

@app.get("/users/me/datasets/{dataset_id}/", response_model=Dataset)
async def get_dataset(
    dataset: Dataset = Depends(get_dataset_or_404)
):
    return dataset

@app.put("/users/me/datasets/{dataset_id}/", response_model=Dataset)
async def update_dataset(
    dataset_id: uuid.UUID,
    dataset_update: DatasetUpdate,
    current_user: User = Depends(auth.get_current_active_user),
    user_datasets: List[Dataset] = Depends(get_user_datasets)
):
    user_id_str = str(current_user.user_id)
    
    dataset_idx = -1
    for i, ds in enumerate(user_datasets):
        if ds.dataset_id == dataset_id:
            dataset_idx = i
            break
            
    if dataset_idx == -1:
        logger.warning(f"Attempted to update non-existent dataset {dataset_id} for user {user_id_str}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    existing_dataset = user_datasets[dataset_idx]

    if dataset_update.name is not None and dataset_update.name != existing_dataset.name:
        if any(ds.name == dataset_update.name for ds in user_datasets if ds.dataset_id != dataset_id):
            logger.warning(f"Attempted to rename dataset {dataset_id} to a duplicate name \'{dataset_update.name}\' for user {user_id_str}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset with name \'{dataset_update.name}\' already exists."
            )
    
    update_data = dataset_update.model_dump(exclude_unset=True)
    updated_dataset = existing_dataset.model_copy(update=update_data)
    updated_dataset.updated_at = datetime.utcnow()
    
    all_datasets_meta = _read_datasets_meta()
    all_datasets_meta[user_id_str][dataset_idx] = updated_dataset
    _write_datasets_meta(all_datasets_meta)
    logger.info(f"Dataset {dataset_id} updated for user {user_id_str}")
    return updated_dataset

@app.delete("/users/me/datasets/{dataset_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user),
    user_datasets: List[Dataset] = Depends(get_user_datasets)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)
    
    dataset_to_delete = next((ds for ds in user_datasets if ds.dataset_id == dataset_id), None)
    
    if not dataset_to_delete:
        logger.warning(f"Attempted to delete non-existent dataset {dataset_id} for user {user_id_str}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
        
    all_datasets_meta = _read_datasets_meta()
    all_datasets_meta[user_id_str] = [ds for ds in user_datasets if ds.dataset_id != dataset_id]
    if not all_datasets_meta[user_id_str]:
        all_datasets_meta.pop(user_id_str, None)
    _write_datasets_meta(all_datasets_meta)
    logger.info(f"Dataset {dataset_id} deleted from metadata for user {user_id_str}")

    datasources_meta = _read_datasources_meta()
    user_datasources = datasources_meta.get(user_id_str, {})
    
    if dataset_id_str in user_datasources:
        files_in_dataset = user_datasources.pop(dataset_id_str, [])
        if not user_datasources:
            datasources_meta.pop(user_id_str, None)
        else:
            datasources_meta[user_id_str] = user_datasources
        _write_datasources_meta(datasources_meta)
        logger.info(f"Datasources for dataset {dataset_id} deleted from metadata for user {user_id_str}")

        for file_meta in files_in_dataset:
            try:
                vector_store_manager.delete_documents(
                    user_id=user_id_str,
                    data_source_id=file_meta.data_source_id,
                    dataset_id=dataset_id_str # dataset_idを渡すように修正
                )
                logger.info(f"Deleted document {file_meta.data_source_id} from vector store for user {user_id_str}")
            except Exception as e:
                logger.error(f"Error deleting {file_meta.data_source_id} from vector store for user {user_id_str}: {e}", exc_info=True)
                
    return


@app.post("/documents/upload/", response_model=DataSourceMeta, deprecated=True, summary="Deprecated: Use upload to dataset instead")
async def upload_document_deprecated(
    current_user: User = Depends(auth.get_current_active_user),
    file: UploadFile = File(...),
    embedding_strategy: Optional[str] = Form(None),
    chunking_strategy: Optional[str] = Form(None),
    chunking_params_json: Optional[str] = Form(None)
):
    logger.warning(f"Deprecated endpoint /documents/upload/ called by user {current_user.user_id}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="This endpoint is deprecated. Please use /users/me/datasets/{dataset_id}/documents/upload/."
    )


@app.post("/users/me/datasets/{dataset_id}/documents/upload/", response_model=DataSourceMeta)
async def upload_document_to_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user),
    dataset: Dataset = Depends(get_dataset_or_404),
    file: UploadFile = File(...),
    embedding_strategy: Optional[str] = Form(None),
    chunking_strategy: Optional[str] = Form(None),
    chunking_params_json: Optional[str] = Form(None)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)

    original_filename = file.filename if file.filename else "unknown_file"
    file_extension = Path(original_filename).suffix.lstrip(".").lower()

    if file_extension not in SUPPORTED_FILE_TYPES.__args__:
        logger.warning(f"Unsupported file type \'{file_extension}\' uploaded by user {user_id_str} to dataset {dataset_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: \'{file_extension}\'. Supported types are: {\', \'.join(SUPPORTED_FILE_TYPES.__args__)}"
        )

    data_source_id = f"{original_filename}_{uuid.uuid4().hex[:8]}"
    temp_file_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_filename}"

    emb_strategy_name: str = embedding_strategy or embedding_manager.default_strategy_name
    if not emb_strategy_name:
        logger.error("Embedding strategies not properly configured.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Embedding strategies not properly configured.")

    chk_strategy_name: str = chunking_strategy or chunking_manager.default_strategy_name
    if not chk_strategy_name:
        logger.error("Chunking strategies not properly configured.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Chunking strategies not properly configured.")

    chk_params: Dict[str, Any] = {}
    if chunking_params_json:
        try:
            chk_params = json.loads(chunking_params_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON format for chunking_params_json from user {user_id_str}: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format for chunking_params_json.")

    if emb_strategy_name not in embedding_manager.get_available_strategies():
        logger.warning(f"Invalid embedding strategy \'{emb_strategy_name}\' provided by user {user_id_str}")
        raise HTTPException(status_code=400, detail=f"Invalid embedding strategy: {emb_strategy_name}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Temporary file saved to {temp_file_path}")
        
        effective_chunk_params: Dict[str, Any] = chk_params or {}
        chunk_size_to_use: int = effective_chunk_params.get("chunk_size", 1000) 
        chunk_overlap_to_use: int = effective_chunk_params.get("chunk_overlap", 200)

        split_docs_to_process: List[Document] = load_and_split_document(
            file_path=str(temp_file_path),
            file_type=file_extension,
            chunk_size=chunk_size_to_use,
            chunk_overlap=chunk_overlap_to_use
        )
        
        if not split_docs_to_process:
             logger.warning(f"Document {original_filename} is empty or could not be processed for user {user_id_str}")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document is empty or could not be loaded/split into processable chunks.")

        num_added_to_vsm: int = vector_store_manager.add_documents(
            user_id=user_id_str,
            data_source_id=data_source_id, 
            documents=split_docs_to_process,
            embedding_strategy_name=emb_strategy_name,
            chunking_strategy_name=chk_strategy_name, 
            chunking_params=effective_chunk_params,
            dataset_id=dataset_id_str
        )
        logger.info(f"Added {num_added_to_vsm} chunks to vector store for file {data_source_id} (user {user_id_str}, dataset {dataset_id})")

        if num_added_to_vsm == 0 and split_docs_to_process:
            logger.error(f"Failed to add document chunks to vector store for file {data_source_id} (user {user_id_str})")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add document chunks to vector store after processing.")

        final_chunking_strategy_instance = chunking_manager.get_strategy(chk_strategy_name, params=effective_chunk_params)
        final_chunking_config_recorded: Dict[str, Any] = final_chunking_strategy_instance.get_config()

        datasources_meta_storage: Dict[str, Dict[str, List[DataSourceMeta]]] = _read_datasources_meta()
        user_specific_datasources: Dict[str, List[DataSourceMeta]] = datasources_meta_storage.get(user_id_str, {})
        dataset_specific_files: List[DataSourceMeta] = user_specific_datasources.get(dataset_id_str, [])
        
        new_meta = DataSourceMeta(
            data_source_id=data_source_id,
            dataset_id=dataset_id,
            original_filename=original_filename,
            status="processed",
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=len(split_docs_to_process),
            embedding_strategy_used=emb_strategy_name,
            chunking_strategy_used=final_chunking_strategy_instance.get_name(),
            chunking_config_used=final_chunking_config_recorded
        )
        dataset_specific_files.append(new_meta)
        user_specific_datasources[dataset_id_str] = user_specific_datasources.get(dataset_id_str, []) + [new_meta]
        datasources_meta_storage[user_id_str] = user_specific_datasources
        _write_datasources_meta(datasources_meta_storage)
        logger.info(f"Metadata for file {data_source_id} recorded for user {user_id_str}, dataset {dataset_id}")

        return new_meta
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during document upload for user {user_id_str}, dataset {dataset_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred processing file: {str(e)}")
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {temp_file_path}: {e}")

@app.get("/users/me/datasets/{dataset_id}/documents/", response_model=List[DataSourceMeta])
async def list_documents_in_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user),
    dataset: Dataset = Depends(get_dataset_or_404),
    user_datasources: Dict[str, Dict[str, List[DataSourceMeta]]] = Depends(get_user_datasources)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)

    dataset_specific_files: List[DataSourceMeta] = user_datasources.get(dataset_id_str, [])
    return dataset_specific_files

@app.delete("/users/me/datasets/{dataset_id}/documents/{data_source_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_from_dataset(
    dataset_id: uuid.UUID,
    data_source_id: str, 
    current_user: User = Depends(auth.get_current_active_user),
    dataset: Dataset = Depends(get_dataset_or_404),
    user_datasources: Dict[str, Dict[str, List[DataSourceMeta]]] = Depends(get_user_datasources)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)

    if dataset_id_str not in user_datasources:
        logger.warning(f"Dataset {dataset_id} not found in file metadata for user {user_id_str}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset (in file metadata) not found.")
    
    files_in_dataset: List[DataSourceMeta] = user_datasources[dataset_id_str]
    file_to_delete_idx: int = -1
    for i, file_meta in enumerate(files_in_dataset):
        if file_meta.data_source_id == data_source_id:
            file_to_delete_idx = i
            break
            
    if file_to_delete_idx == -1:
        logger.warning(f"File {data_source_id} not found in dataset {dataset_id} for user {user_id_str}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found in the specified dataset.")

    files_in_dataset.pop(file_to_delete_idx)
    all_datasources_meta: Dict[str, Dict[str, List[DataSourceMeta]]] = _read_datasources_meta()
    if not files_in_dataset: 
        all_datasources_meta[user_id_str].pop(dataset_id_str)
        if not all_datasources_meta[user_id_str]: 
             all_datasources_meta.pop(user_id_str, None)
    _write_datasources_meta(all_datasources_meta)
    logger.info(f"File {data_source_id} deleted from metadata for user {user_id_str}, dataset {dataset_id}")

    try:
        vector_store_manager.delete_documents(
            user_id=user_id_str,
            data_source_id=data_source_id, 
            dataset_id=dataset_id_str 
        )
        logger.info(f"Deleted document {data_source_id} from vector store for user {user_id_str}, dataset {dataset_id}")
    except Exception as e:
        logger.error(f"Error deleting file {data_source_id} (dataset {dataset_id_str}) from vector store for user {user_id_str}: {e}", exc_info=True)
        pass 
    return

@app.get("/documents/", response_model=List[DataSourceMeta], deprecated=True, summary="Deprecated: Use GET /users/me/datasets/{dataset_id}/documents/ instead.")
async def list_documents_deprecated(
    current_user: User = Depends(auth.get_current_active_user)
):
    logger.warning(f"Deprecated endpoint /documents/ called by user {current_user.user_id}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="This endpoint is deprecated. Please use /users/me/datasets/{dataset_id}/documents/ to list files within a specific dataset."
    )

@app.post("/chat/query/", response_model=ChatQueryResponse)
async def query_rag_chat(
    query_request: ChatQueryRequest,
    current_user: User = Depends(auth.get_current_active_user),
    all_user_files_by_dataset: Dict[str, Dict[str, List[DataSourceMeta]]] = Depends(get_user_datasources)
):
    user_id_str: str = str(current_user.user_id)
    selected_rag_strategy: str = query_request.rag_strategy
    if not rag_strategy_factory.is_strategy_available(selected_rag_strategy):
        logger.warning(f"Invalid RAG strategy \'{selected_rag_strategy}\' provided by user {user_id_str}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RAG strategy: \'{selected_rag_strategy}\'. Available strategies are: {', '.join(rag_strategy_factory.get_available_strategies())}"
        )

    # RAG戦略の設定を読み込む (例: グローバル設定から)
    # ここでは仮に空のdictを渡すが、実際にはconfig.jsonなどから読み込む
    rag_strategy_config = {}

    try:
        # ファクトリからRAG戦略のインスタンスを取得
        rag_strategy_instance = rag_strategy_factory.get_strategy(selected_rag_strategy, rag_strategy_config)

        # 関連するデータソースIDの特定ロジックは現状維持
        target_data_source_ids_for_query: List[str] = []

        # embedding_strategy_for_retrievalをここで決定
        embedding_strategy_for_retrieval: str = embedding_manager.get_available_strategies()[0]

        if query_request.data_source_ids:
            target_data_source_ids_for_query.extend(query_request.data_source_ids)
            if target_data_source_ids_for_query:
                first_file_id_to_check: str = target_data_source_ids_for_query[0]
                meta_found: Optional[DataSourceMeta] = None
                # Iterate through all datasets to find the data_source_id
                for dataset_files in all_user_files_by_dataset.values():
                    meta_found = next((meta for meta in dataset_files.values() if meta.data_source_id == first_file_id_to_check), None)
                    if meta_found: break
                if meta_found and meta_found.embedding_strategy_used:
                    embedding_strategy_for_retrieval = meta_found.embedding_strategy_used
                elif not meta_found:
                     logger.warning(f"Specified data_source_id {first_file_id_to_check} not found for user {user_id_str}")
                     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Specified data_source_id {first_file_id_to_check} not found.")
        
        elif query_request.dataset_ids:
            first_strategy_set: bool = False
            for dataset_uuid in query_request.dataset_ids:
                dataset_str_id: str = str(dataset_uuid)
                # Check if the dataset belongs to the user (optional, as get_user_datasources already filters)
                user_actual_datasets: List[Dataset] = _read_datasets_meta().get(user_id_str, [])
                if not any(ds.dataset_id == dataset_uuid for ds in user_actual_datasets):
                    logger.warning(f"Access to dataset {dataset_str_id} is forbidden or dataset not found for user {user_id_str}. Proceeding if files exist.")
                    pass

                files_in_dataset: List[DataSourceMeta] = all_user_files_by_dataset.get(dataset_str_id, [])
                for file_meta in files_in_dataset:
                    target_data_source_ids_for_query.append(file_meta.data_source_id)
                    if not first_strategy_set and file_meta.embedding_strategy_used:
                        embedding_strategy_for_retrieval = file_meta.embedding_strategy_used
                        first_strategy_set = True
        else:
            # Default: Query all documents from all datasets of the user
            first_strategy_set = False
            for dataset_files in all_user_files_by_dataset.values():                for file_meta in dataset_files.values():.values(): # ここを修正
                    target_data_source_ids_for_query.append(file_meta.data_source_id)
                    if not first_strategy_set and file_meta.embedding_strategy_used:
                        embedding_strategy_for_retrieval = file_meta.embedding_strategy_used
                        first_strategy_set = True

        if not target_data_source_ids_for_query:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for the query based on the provided criteria.")

        # RAG戦略を実行
        answer_data: Dict[str, Any] = rag_strategy_instance.execute(
            user_id=user_id_str,
            question=query_request.question,
            data_source_ids=list(set(target_data_source_ids_for_query)),
            embedding_strategy_for_retrieval=embedding_strategy_for_retrieval
        )


