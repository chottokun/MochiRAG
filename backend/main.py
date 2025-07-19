import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm

from . import auth
from .models import (
    User, UserCreate, Token, DataSourceMeta,
    ChatQueryRequest, ChatQueryResponse, # DocumentUploadRequest は削除 (フォームパラメータで受け取るため)
    Dataset, DatasetCreate, DatasetUpdate # 追加: データセットモデル
)

# Core module imports - assuming they are structured to be importable
# and global manager instances are available.
try:
    from core.document_processor import SUPPORTED_FILE_TYPES, load_and_split_document # load_and_split_document は VSM内部で呼ばれる想定
    from core.vector_store_manager import vector_store_manager
    from core.embedding_manager import embedding_manager
    from core.chunking_manager import chunking_manager
    from core.rag_chain import get_rag_response, RAG_STRATEGY_TYPE, AVAILABLE_RAG_STRATEGIES
    from langchain_core.documents import Document # Document型を直接使うため
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
        from core.rag_chain import get_rag_response, RAG_STRATEGY_TYPE, AVAILABLE_RAG_STRATEGIES
        from langchain_core.documents import Document
    except ImportError:
        raise ImportError(f"Could not import core modules. Ensure 'core' is in PYTHONPATH and managers are initialized. Original error: {e}")


app = FastAPI()

# --- Metadata Store Configuration ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_META_PATH = DATA_DIR / "datasets_meta.json" # For Dataset objects
DATASOURCES_META_PATH = DATA_DIR / "datasources_meta.json" # For DataSourceMeta objects (file metadata)
TMP_UPLOAD_DIR = DATA_DIR / "tmp_uploads"
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper functions for Dataset Metadata ---
def _ensure_datasets_meta_file_exists():
    if not DATASETS_META_PATH.exists():
        with open(DATASETS_META_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f) # User ID will be the key, value is a list of Dataset objects

def _read_datasets_meta() -> Dict[str, List[Dataset]]:
    _ensure_datasets_meta_file_exists()
    with open(DATASETS_META_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            parsed_data: Dict[str, List[Dataset]] = {}
            for user_id_str, datasets_json_list in data.items():
                parsed_data[user_id_str] = [Dataset(**ds_json) for ds_json in datasets_json_list]
            return parsed_data
        except json.JSONDecodeError:
            return {}

def _write_datasets_meta(data: Dict[str, List[Dataset]]):
    _ensure_datasets_meta_file_exists()
    serializable_data: Dict[str, List[Dict[str, Any]]] = {}
    for user_id_str, datasets_list in data.items():
        serializable_data[user_id_str] = [ds.model_dump(mode="json") for ds in datasets_list]
    with open(DATASETS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)

# --- Helper functions for DataSource (File) Metadata ---
def _ensure_datasources_meta_file_exists():
    if not DATASOURCES_META_PATH.exists():
        with open(DATASOURCES_META_PATH, "w", encoding="utf-8") as f:
            # New structure: { "user_id": { "dataset_id": [DataSourceMeta, ...], ... }, ... }
            json.dump({}, f)

def _read_datasources_meta() -> Dict[str, Dict[str, List[DataSourceMeta]]]:
    _ensure_datasources_meta_file_exists()
    with open(DATASOURCES_META_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            parsed_data: Dict[str, Dict[str, List[DataSourceMeta]]] = {}
            for user_id_str, user_datasets_json in data.items():
                parsed_data[user_id_str] = {}
                for dataset_id_str, ds_meta_list_json in user_datasets_json.items():
                    parsed_data[user_id_str][dataset_id_str] = [
                        DataSourceMeta(**meta) for meta in ds_meta_list_json
                    ]
            return parsed_data
        except json.JSONDecodeError:
            return {}

def _write_datasources_meta(data: Dict[str, Dict[str, List[DataSourceMeta]]]):
    _ensure_datasources_meta_file_exists()
    serializable_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for user_id_str, user_datasets_pydantic in data.items():
        serializable_data[user_id_str] = {}
        for dataset_id_str, ds_meta_list_pydantic in user_datasets_pydantic.items():
            serializable_data[user_id_str][dataset_id_str] = [
                meta.model_dump(mode="json") for meta in ds_meta_list_pydantic
            ]
    with open(DATASOURCES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)

# Initialize metadata files on startup
_ensure_datasets_meta_file_exists()
_ensure_datasources_meta_file_exists()

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
async def create_user_endpoint(user_data: UserCreate): # Renamed from create_user to avoid conflict
    try:
        created_user_in_db = auth.create_user_in_db(user_data)
        return User(**created_user_in_db.model_dump())
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(auth.get_current_active_user)):
    return current_user

# --- Dataset Endpoints ---

@app.post("/users/me/datasets/", response_model=Dataset, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset_create: DatasetCreate,
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    datasets_meta = _read_datasets_meta()
    user_datasets = datasets_meta.get(user_id_str, [])

    # Check for duplicate dataset name for the same user
    if any(ds.name == dataset_create.name for ds in user_datasets):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset with name '{dataset_create.name}' already exists."
        )

    new_dataset = Dataset(
        user_id=current_user.user_id,
        name=dataset_create.name,
        description=dataset_create.description
        # dataset_id, created_at, updated_at are auto-generated
    )
    user_datasets.append(new_dataset)
    datasets_meta[user_id_str] = user_datasets
    _write_datasets_meta(datasets_meta)
    return new_dataset

@app.get("/users/me/datasets/", response_model=List[Dataset])
async def list_datasets(current_user: User = Depends(auth.get_current_active_user)):
    user_id_str = str(current_user.user_id)
    datasets_meta = _read_datasets_meta()
    return datasets_meta.get(user_id_str, [])

@app.get("/users/me/datasets/{dataset_id}/", response_model=Dataset)
async def get_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    datasets_meta = _read_datasets_meta()
    user_datasets = datasets_meta.get(user_id_str, [])
    
    dataset = next((ds for ds in user_datasets if ds.dataset_id == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset

@app.put("/users/me/datasets/{dataset_id}/", response_model=Dataset)
async def update_dataset(
    dataset_id: uuid.UUID,
    dataset_update: DatasetUpdate,
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    datasets_meta = _read_datasets_meta()
    user_datasets = datasets_meta.get(user_id_str, [])
    
    dataset_idx = -1
    for i, ds in enumerate(user_datasets):
        if ds.dataset_id == dataset_id:
            dataset_idx = i
            break
            
    if dataset_idx == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    existing_dataset = user_datasets[dataset_idx]

    # Check for duplicate name if name is being changed
    if dataset_update.name is not None and dataset_update.name != existing_dataset.name:
        if any(ds.name == dataset_update.name for ds in user_datasets if ds.dataset_id != dataset_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset with name '{dataset_update.name}' already exists."
            )
    
    update_data = dataset_update.model_dump(exclude_unset=True)
    updated_dataset = existing_dataset.model_copy(update=update_data)
    updated_dataset.updated_at = datetime.utcnow()
    
    user_datasets[dataset_idx] = updated_dataset
    datasets_meta[user_id_str] = user_datasets
    _write_datasets_meta(datasets_meta)
    return updated_dataset

@app.delete("/users/me/datasets/{dataset_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)
    
    # Delete from datasets_meta.json
    datasets_meta = _read_datasets_meta()
    user_datasets = datasets_meta.get(user_id_str, [])
    dataset_to_delete = next((ds for ds in user_datasets if ds.dataset_id == dataset_id), None)
    
    if not dataset_to_delete:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
        
    user_datasets = [ds for ds in user_datasets if ds.dataset_id != dataset_id]
    if user_datasets:
        datasets_meta[user_id_str] = user_datasets
    else:
        datasets_meta.pop(user_id_str, None) # Remove user entry if no datasets left
    _write_datasets_meta(datasets_meta)
    
    # Delete associated files from datasources_meta.json and vector store
    datasources_meta = _read_datasources_meta()
    user_datasources = datasources_meta.get(user_id_str, {})
    
    if dataset_id_str in user_datasources:
        files_in_dataset = user_datasources.pop(dataset_id_str, [])
        if not user_datasources: # if no other datasets for this user
            datasources_meta.pop(user_id_str, None)
        else:
            datasources_meta[user_id_str] = user_datasources
        _write_datasources_meta(datasources_meta)

        # Delete documents from vector store for each file in the dataset
        for file_meta in files_in_dataset:
            try:
                # Assuming vector_store_manager can delete by user_id and data_source_id
                # We need to pass dataset_id as well if the VSM expects it for namespacing
                vector_store_manager.delete_documents(
                    user_id=user_id_str,
                    data_source_id=file_meta.data_source_id
                    # dataset_id=dataset_id_str # This might be needed depending on VSM implementation
                )
            except Exception as e:
                # Log error, but continue deletion process
                # Consider how to handle partial failures (e.g., if VSM deletion fails)
                print(f"Error deleting {file_meta.data_source_id} from vector store: {e}")
                # Potentially re-raise or collect errors to return to user if critical

    return # FastAPI handles 204 No Content response automatically


# --- Document/File Endpoints (Now dataset-specific) ---

@app.post("/documents/upload/", response_model=DataSourceMeta, deprecated=True, summary="Deprecated: Use upload to dataset instead")
async def upload_document_deprecated(
    current_user: User = Depends(auth.get_current_active_user),
    file: UploadFile = File(...),
    embedding_strategy: Optional[str] = Form(None),
    chunking_strategy: Optional[str] = Form(None),
    chunking_params_json: Optional[str] = Form(None)
):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="This endpoint is deprecated. Please use /users/me/datasets/{dataset_id}/documents/upload/."
    )


@app.post("/users/me/datasets/{dataset_id}/documents/upload/", response_model=List[DataSourceMeta])
async def upload_documents_to_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user),
    files: List[UploadFile] = File(...),
    embedding_strategy: Optional[str] = Form(None),
    chunking_strategy: Optional[str] = Form(None),
    chunking_params_json: Optional[str] = Form(None)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)

    # Verify dataset exists and belongs to user
    all_datasets_meta = _read_datasets_meta()
    user_owned_datasets = all_datasets_meta.get(user_id_str, [])
    if not any(ds.dataset_id == dataset_id for ds in user_owned_datasets):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    datasources_meta_storage = _read_datasources_meta()
    user_specific_datasources = datasources_meta_storage.get(user_id_str, {})
    dataset_specific_files = user_specific_datasources.get(dataset_id_str, [])

    processed_files_meta = []

    for file in files:
        original_filename = file.filename if file.filename else "unknown_file"
        file_extension = Path(original_filename).suffix.lstrip('.').lower()

        if file_extension not in SUPPORTED_FILE_TYPES.__args__: # type: ignore
            # Skip unsupported files, or collect errors to report at the end
            print(f"Skipping unsupported file type: {original_filename}")
            continue

        data_source_id = f"{original_filename}_{uuid.uuid4().hex[:8]}" # Unique ID for this file
        temp_file_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_filename}"

        emb_strategy_name = embedding_strategy or embedding_manager.default_strategy_name
        if not emb_strategy_name:
            raise HTTPException(status_code=503, detail="Service Unavailable: Embedding strategies not properly configured.")

        chk_strategy_name = chunking_strategy or chunking_manager.default_strategy_name
        if not chk_strategy_name:
            raise HTTPException(status_code=503, detail="Service Unavailable: Chunking strategies not properly configured.")

        chk_params: Optional[Dict[str, Any]] = None
        if chunking_params_json:
            try:
                chk_params = json.loads(chunking_params_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for chunking_params_json.")

        if emb_strategy_name not in embedding_manager.get_available_strategies():
            raise HTTPException(status_code=400, detail=f"Invalid embedding strategy: {emb_strategy_name}")

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            effective_chunk_params = chk_params or {}
            chunk_size_to_use = effective_chunk_params.get("chunk_size", 1000)
            chunk_overlap_to_use = effective_chunk_params.get("chunk_overlap", 200)

            split_docs_to_process = load_and_split_document(
                file_path=str(temp_file_path),
                file_type=file_extension, # type: ignore
                chunk_size=chunk_size_to_use,
                chunk_overlap=chunk_overlap_to_use
            )

            if not split_docs_to_process:
                 if temp_file_path.exists(): temp_file_path.unlink()
                 print(f"Skipping empty or unprocessable file: {original_filename}")
                 continue

            num_added_to_vsm = vector_store_manager.add_documents(
                user_id=user_id_str,
                data_source_id=data_source_id,
                documents=split_docs_to_process,
                embedding_strategy_name=emb_strategy_name,
                chunking_strategy_name=chk_strategy_name,
                chunking_params=effective_chunk_params,
                dataset_id=dataset_id_str
            )

            if num_added_to_vsm == 0 and split_docs_to_process:
                if temp_file_path.exists(): temp_file_path.unlink()
                print(f"Failed to add chunks to vector store for file: {original_filename}")
                continue

            final_chunking_strategy_instance = chunking_manager.get_strategy(chk_strategy_name, params=effective_chunk_params)
            final_chunking_config_recorded = final_chunking_strategy_instance.get_config()

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
            processed_files_meta.append(new_meta)

        except Exception as e:
            print(f"Error processing file {original_filename}: {e}")
            # Decide if one failure should stop the whole batch
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()
            file.file.close()

    if processed_files_meta:
        user_specific_datasources[dataset_id_str] = dataset_specific_files
        datasources_meta_storage[user_id_str] = user_specific_datasources
        _write_datasources_meta(datasources_meta_storage)

    return processed_files_meta

@app.get("/users/me/datasets/{dataset_id}/documents/", response_model=List[DataSourceMeta])
async def list_documents_in_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)

    all_datasets_meta = _read_datasets_meta()
    user_owned_datasets = all_datasets_meta.get(user_id_str, [])
    if not any(ds.dataset_id == dataset_id for ds in user_owned_datasets):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    datasources_meta = _read_datasources_meta()
    user_specific_datasources = datasources_meta.get(user_id_str, {})
    dataset_specific_files = user_specific_datasources.get(dataset_id_str, [])
    return dataset_specific_files

@app.delete("/users/me/datasets/{dataset_id}/documents/{data_source_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_from_dataset(
    dataset_id: uuid.UUID,
    data_source_id: str, 
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    dataset_id_str = str(dataset_id)

    all_datasets_meta = _read_datasets_meta()
    user_owned_datasets = all_datasets_meta.get(user_id_str, [])
    if not any(ds.dataset_id == dataset_id for ds in user_owned_datasets):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    datasources_meta_storage = _read_datasources_meta()
    user_datasources = datasources_meta_storage.get(user_id_str, {})
    if dataset_id_str not in user_datasources:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset (in file metadata) not found.")
    
    files_in_dataset = user_datasources[dataset_id_str]
    file_to_delete_idx = -1
    for i, file_meta in enumerate(files_in_dataset):
        if file_meta.data_source_id == data_source_id:
            file_to_delete_idx = i
            break
            
    if file_to_delete_idx == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found in the specified dataset.")

    files_in_dataset.pop(file_to_delete_idx)
    if not files_in_dataset: 
        user_datasources.pop(dataset_id_str)
        if not user_datasources: 
             datasources_meta_storage.pop(user_id_str, None)
    _write_datasources_meta(datasources_meta_storage)

    try:
        vector_store_manager.delete_documents(
            user_id=user_id_str,
            data_source_id=data_source_id, 
            dataset_id=dataset_id_str 
        )
    except Exception as e:
        print(f"Error deleting file {data_source_id} (dataset {dataset_id_str}) from vector store: {e}")
        pass 
    return

@app.get("/documents/", response_model=List[DataSourceMeta], deprecated=True, summary="Deprecated: Use GET /users/me/datasets/{dataset_id}/documents/ instead.")
async def list_documents_deprecated(current_user: User = Depends(auth.get_current_active_user)):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="This endpoint is deprecated. Please use /users/me/datasets/{dataset_id}/documents/ to list files within a specific dataset."
    )

@app.post("/chat/query/", response_model=ChatQueryResponse)
async def query_rag_chat(
    query_request: ChatQueryRequest,
    current_user: User = Depends(auth.get_current_active_user)
):
    user_id_str = str(current_user.user_id)
    selected_rag_strategy = query_request.rag_strategy
    if selected_rag_strategy not in AVAILABLE_RAG_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RAG strategy: '{selected_rag_strategy}'. Available strategies are: {', '.join(AVAILABLE_RAG_STRATEGIES)}"
        )

    embedding_strategy_for_retrieval = embedding_manager.get_available_strategies()[0] 
    target_data_source_ids_for_query: List[str] = []
    
    all_user_files_by_dataset = _read_datasources_meta().get(user_id_str, {})

    if query_request.data_source_ids:
        target_data_source_ids_for_query.extend(query_request.data_source_ids)
        # Determine embedding strategy from the first specified data_source_id
        if target_data_source_ids_for_query:
            first_file_id_to_check = target_data_source_ids_for_query[0]
            meta_found = None
            for dataset_files in all_user_files_by_dataset.values():
                meta_found = next((meta for meta in dataset_files if meta.data_source_id == first_file_id_to_check), None)
                if meta_found:
                    break
            if meta_found and meta_found.embedding_strategy_used:
                embedding_strategy_for_retrieval = meta_found.embedding_strategy_used
            elif not meta_found:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Specified data_source_id {first_file_id_to_check} not found.")

    elif query_request.dataset_ids is not None:
        # 空リストなら何も対象にしない（404）
        if isinstance(query_request.dataset_ids, list) and len(query_request.dataset_ids) == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No dataset selected for query.")
        first_strategy_set = False
        for dataset_uuid in query_request.dataset_ids:
            dataset_str_id = str(dataset_uuid)
            # Ensure the queried dataset belongs to the user
            user_actual_datasets = _read_datasets_meta().get(user_id_str, [])
            if not any(ds.dataset_id == dataset_uuid for ds in user_actual_datasets):
                pass
            files_in_dataset = all_user_files_by_dataset.get(dataset_str_id, [])
            for file_meta in files_in_dataset:
                target_data_source_ids_for_query.append(file_meta.data_source_id)
                if not first_strategy_set and file_meta.embedding_strategy_used:
                    embedding_strategy_for_retrieval = file_meta.embedding_strategy_used
                    first_strategy_set = True
    else:
        # Default: Query all documents from all datasets of the user
        first_strategy_set = False
        for dataset_files in all_user_files_by_dataset.values():
            for file_meta in dataset_files:
                target_data_source_ids_for_query.append(file_meta.data_source_id)
                if not first_strategy_set and file_meta.embedding_strategy_used:
                    embedding_strategy_for_retrieval = file_meta.embedding_strategy_used
                    first_strategy_set = True
    
    if not target_data_source_ids_for_query:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for the query based on the provided criteria.")

    try:
        answer_data = get_rag_response(
            user_id=user_id_str,
            question=query_request.question,
            data_source_ids=list(set(target_data_source_ids_for_query)), # Ensure unique IDs
            rag_strategy=selected_rag_strategy, # type: ignore
            embedding_strategy_for_retrieval=embedding_strategy_for_retrieval
        )

        processed_sources = None
        if answer_data.get("sources"):
            processed_sources = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in answer_data["sources"]
            ]

        return ChatQueryResponse(
            answer=answer_data["answer"],
            strategy_used=selected_rag_strategy,
            sources=processed_sources
        )
    except Exception as e:
        # logger.error(f"Error in /chat/query/ for user {user_id_str} with strategy {selected_rag_strategy}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred processing your query with strategy '{selected_rag_strategy}': {str(e)}"
        )

# To run this app (for testing locally):
# Ensure you are in the project root directory (/app)
# uvicorn backend.main:app --reload --port 8000
# (The sys.path hack for core modules helps with this type of execution)
