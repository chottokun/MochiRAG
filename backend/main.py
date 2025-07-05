import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm

from . import auth
from .models import User, UserCreate, Token, UserInDB, DataSourceMeta, ChatQueryRequest, ChatQueryResponse # Added request/response models

# Assuming core modules are in the parent directory of backend, or installed
# This might need adjustment based on final project structure and how it's run.
# For now, direct import assumes PYTHONPATH is set up or they are discoverable.
try:
    from core.document_processor import load_and_split_document, SUPPORTED_FILE_TYPES
    from core.vector_store import add_documents_to_vector_db
    from core.rag_chain import get_rag_response, RAG_STRATEGY_TYPE, AVAILABLE_RAG_STRATEGIES # Import strategy type
except ImportError as e:
    # This provides a fallback for environments where 'core' is not directly in PYTHONPATH
    # e.g. when running 'uvicorn backend.main:app' from project root.
    import sys
    # Add the project root to sys.path if 'core' is not found.
    # This assumes the script is run from a context where 'app' (project root) is parent of 'backend'
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from core.document_processor import load_and_split_document, SUPPORTED_FILE_TYPES
        from core.vector_store import add_documents_to_vector_db
        from core.rag_chain import get_rag_response, RAG_STRATEGY_TYPE, AVAILABLE_RAG_STRATEGIES # Import strategy type
    except ImportError:
        raise ImportError(f"Could not import core modules. Ensure 'core' is in PYTHONPATH. Original error: {e}")


app = FastAPI()

# --- Metadata Store for Uploaded Documents ---
DATASOURCES_META_PATH = Path(__file__).resolve().parent.parent / "data" / "datasources_meta.json"
TMP_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "tmp_uploads"
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

def _ensure_metadata_file_exists():
    if not DATASOURCES_META_PATH.exists():
        with open(DATASOURCES_META_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)

def _read_datasources_meta() -> Dict[str, List[DataSourceMeta]]:
    _ensure_metadata_file_exists()
    with open(DATASOURCES_META_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Convert list of dicts to list of DataSourceMeta objects
            # Pydantic can parse this automatically if the structure matches
            parsed_data: Dict[str, List[DataSourceMeta]] = {}
            for user_id, meta_list_json in data.items():
                parsed_data[user_id] = [DataSourceMeta(**meta) for meta in meta_list_json]
            return parsed_data
        except json.JSONDecodeError:
            return {} # Return empty if file is corrupted or empty

def _write_datasources_meta(data: Dict[str, List[DataSourceMeta]]):
    _ensure_metadata_file_exists()
    # Convert DataSourceMeta objects back to dicts for JSON serialization
    serializable_data: Dict[str, List[Dict[str, Any]]] = {}
    for user_id, meta_list_pydantic in data.items():
        serializable_data[user_id] = [meta.model_dump() for meta in meta_list_pydantic]

    with open(DATASOURCES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)

# Initialize metadata file on startup (useful for the first run)
_ensure_metadata_file_exists()

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

@app.post("/documents/upload/", response_model=DataSourceMeta)
async def upload_document(
    current_user: User = Depends(auth.get_current_active_user),
    file: UploadFile = File(...)
):
    original_filename = file.filename if file.filename else "unknown_file"
    file_extension = Path(original_filename).suffix.lstrip('.').lower()

    if file_extension not in SUPPORTED_FILE_TYPES.__args__: # Check against Literal values
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{file_extension}'. Supported types are: {', '.join(SUPPORTED_FILE_TYPES.__args__)}"
        )

    # Use a unique ID for the data_source_id to avoid collisions if same filename uploaded multiple times
    # Or, if filename should be unique per user, this could just be original_filename
    data_source_id = f"{original_filename}_{uuid.uuid4().hex[:8]}"

    temp_file_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_filename}"

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the document
        # load_and_split_document expects file_type as 'txt', 'md', 'pdf'
        doc_chunks = load_and_split_document(str(temp_file_path), file_type=file_extension) # type: ignore

        if not doc_chunks:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document is empty or could not be processed into chunks.")

        # Add to vector DB
        add_documents_to_vector_db(
            user_id=str(current_user.user_id), # Ensure user_id is string for Chroma metadata if it's UUID
            data_source_id=data_source_id,
            documents=doc_chunks
        )

        # Store metadata
        metas = _read_datasources_meta()
        user_metas = metas.get(str(current_user.user_id), [])

        new_meta = DataSourceMeta(
            data_source_id=data_source_id,
            original_filename=original_filename,
            status="processed",
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=len(doc_chunks)
        )
        user_metas.append(new_meta)
        metas[str(current_user.user_id)] = user_metas
        _write_datasources_meta(metas)

        return new_meta

    except HTTPException: # Re-raise HTTPException
        raise
    except FileNotFoundError as e: # From load_and_split_document
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e: # From load_and_split_document (e.g. unsupported type, though checked)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Log the full error for debugging
        # logger.error(f"Error during document upload: {e}", exc_info=True) # Needs logger setup
        # For now, generic error
        # Potentially update metadata to "failed" status here
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing file: {e}")
    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        if hasattr(file, 'close'):
             file.file.close() # Ensure the UploadFile stream is closed


@app.get("/documents/", response_model=List[DataSourceMeta])
async def list_documents(current_user: User = Depends(auth.get_current_active_user)):
    metas = _read_datasources_meta()
    user_specific_metas = metas.get(str(current_user.user_id), [])
    return user_specific_metas

@app.post("/chat/query/", response_model=ChatQueryResponse)
async def query_rag_chat(
    query_request: ChatQueryRequest,
    current_user: User = Depends(auth.get_current_active_user)
):
    """
    Handles a chat query, retrieves context from user-specific documents,
    and generates a response using the RAG chain with a specified strategy.
    """
    user_id_str = str(current_user.user_id) # Ensure user_id is a string

    # Validate RAG strategy
    selected_strategy = query_request.rag_strategy
    if selected_strategy not in AVAILABLE_RAG_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RAG strategy: '{selected_strategy}'. Available strategies are: {', '.join(AVAILABLE_RAG_STRATEGIES)}"
        )

    try:
        # Call the RAG processing function from core.rag_chain
        answer = get_rag_response(
            user_id=user_id_str,
            question=query_request.question,
            data_source_ids=query_request.data_source_ids,
            rag_strategy=selected_strategy # Pass the validated strategy
        )

        return ChatQueryResponse(answer=answer, strategy_used=selected_strategy)

    except Exception as e:
        # This is a general fallback. Ideally, specific exceptions from RAG chain
        # (if any beyond LLM connection) would be handled more granularly.
        # logger.error(f"Error in /chat/query/ endpoint for user {user_id_str}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing your query: {str(e)}"
        )


# To run this app (for testing locally):
# Ensure you are in the project root directory (/app)
# uvicorn backend.main:app --reload --port 8000
# (The sys.path hack for core modules helps with this type of execution)
