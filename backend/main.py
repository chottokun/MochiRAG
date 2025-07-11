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
    ChatQueryRequest, ChatQueryResponse # DocumentUploadRequest は削除 (フォームパラメータで受け取るため)
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
    file: UploadFile = File(...),
    embedding_strategy: Optional[str] = Form(None),
    chunking_strategy: Optional[str] = Form(None),
    chunking_params_json: Optional[str] = Form(None) # JSON文字列としてチャンキングパラメータを受け取る
):
    original_filename = file.filename if file.filename else "unknown_file"
    file_extension = Path(original_filename).suffix.lstrip('.').lower()

    if file_extension not in SUPPORTED_FILE_TYPES.__args__:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{file_extension}'. Supported types are: {', '.join(SUPPORTED_FILE_TYPES.__args__)}"
        )

    data_source_id = f"{original_filename}_{uuid.uuid4().hex[:8]}"
    temp_file_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_filename}"

    # 戦略パラメータの処理
    emb_strategy_name = embedding_strategy or embedding_manager.default_strategy_name
    if not emb_strategy_name:
        # EmbeddingManagerが利用可能な戦略をロードできなかった、またはデフォルトが設定できなかった場合
        # （通常は EmbeddingManager の初期化時に警告ログが出ているはず）
        raise HTTPException(status_code=503, detail="Service Unavailable: Embedding strategies not properly configured.")

    chk_strategy_name = chunking_strategy or chunking_manager.default_strategy_name
    if not chk_strategy_name:
        # ChunkingManagerが利用可能な戦略をロードできなかった、またはデフォルトが設定できなかった場合
        raise HTTPException(status_code=503, detail="Service Unavailable: Chunking strategies not properly configured.")

    chk_params: Optional[Dict[str, Any]] = None
    if chunking_params_json:
        try:
            chk_params = json.loads(chunking_params_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for chunking_params_json.")

    # 戦略のバリデーション
    if emb_strategy_name not in embedding_manager.get_available_strategies():
        raise HTTPException(status_code=400, detail=f"Invalid embedding strategy: {emb_strategy_name}")
    # ChunkingManager.get_strategyは存在しない戦略名の場合デフォルトにフォールバックするので、
    # ここでの厳密な存在チェックは必須ではないが、特定の戦略名を期待する場合は追加しても良い。

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Document Loaderを使って一時ファイルからDocumentオブジェクトを作成
        # (load_and_split_document はVectorStoreManager内部で呼ばれるように変更する想定)
        # ここでは、まずファイルの内容を読み込み、Documentオブジェクトを作成する
        # これはVectorStoreManagerの責務の一部になるべき
        # TODO: この部分をVectorStoreManagerに統合する
        docs_to_process = [Document(page_content=temp_file_path.read_text(encoding="utf-8"), metadata={"source": original_filename, "file_path": str(temp_file_path)})]
        # PDFやMarkdownの場合は、適切なローダーを使う必要がある。
        # load_and_split_documentを直接呼び出すか、VSMがそれを行う。
        # ここでは簡略化のため、VSMが元のドキュメントリストを受け付けると仮定。

        num_chunks = vector_store_manager.add_documents(
            user_id=str(current_user.user_id),
            data_source_id=data_source_id,
            documents=docs_to_process, # Langchain Documentのリスト
            embedding_strategy_name=emb_strategy_name,
            chunking_strategy_name=chk_strategy_name,
            chunking_params=chk_params
        )

        if num_chunks == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document is empty or could not be processed into chunks.")

        # 使用されたチャンキング戦略の具体的な設定を取得
        # (get_strategyはインスタンスを返すので、そこからconfigを取得)
        final_chunking_strategy = chunking_manager.get_strategy(chk_strategy_name, params=chk_params)
        final_chunking_config = final_chunking_strategy.get_config()


        metas = _read_datasources_meta()
        user_metas = metas.get(str(current_user.user_id), [])
        new_meta = DataSourceMeta(
            data_source_id=data_source_id,
            original_filename=original_filename,
            status="processed",
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=num_chunks,
            embedding_strategy_used=emb_strategy_name,
            chunking_strategy_used=final_chunking_strategy.get_name(), # パラメータ反映後の名前
            chunking_config_used=final_chunking_config
        )
        user_metas.append(new_meta)
        metas[str(current_user.user_id)] = user_metas
        _write_datasources_meta(metas)

        return new_meta
    except HTTPException:
        raise
    except Exception as e:
        # logger.error(f"Error during document upload: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing file: {str(e)}")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()
        if hasattr(file, 'close') and file.file:
            file.file.close()

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
    user_id_str = str(current_user.user_id)

    selected_rag_strategy = query_request.rag_strategy
    if selected_rag_strategy not in AVAILABLE_RAG_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RAG strategy: '{selected_rag_strategy}'. Available strategies are: {', '.join(AVAILABLE_RAG_STRATEGIES)}"
        )

    # ドキュメントのメタデータを読み込み、使用されたエンベディング戦略を取得する
    # ここでは簡単のため、指定された最初のデータソースIDの戦略を使用する
    # 複数のデータソースが異なる戦略で処理されている場合、より高度な処理が必要
    embedding_strategy_for_retrieval = embedding_manager.get_available_strategies()[0] # デフォルトフォールバック
    if query_request.data_source_ids:
        all_metas = _read_datasources_meta()
        user_metas = all_metas.get(user_id_str, [])
        # 指定された最初のデータソースIDに対応するメタデータを探す
        first_ds_id = query_request.data_source_ids[0]
        meta_found = next((meta for meta in user_metas if meta.data_source_id == first_ds_id), None)
        if meta_found and meta_found.embedding_strategy_used:
            embedding_strategy_for_retrieval = meta_found.embedding_strategy_used
        elif meta_found:
            # embedding_strategy_used が記録されていない古いデータソースかもしれないので警告
            pass # logger.warning(f"DataSource {first_ds_id} has no embedding_strategy_used metadata. Using default.")
        else:
            # logger.warning(f"DataSource {first_ds_id} metadata not found. Using default embedding strategy for retrieval.")
            pass


    try:
        answer = get_rag_response(
            user_id=user_id_str,
            question=query_request.question,
            data_source_ids=query_request.data_source_ids,
            rag_strategy=selected_rag_strategy, # type: ignore
            # retriever_manager と embedding_strategy_for_retrieval は get_rag_response 内部で利用される想定
            # 必要であれば、get_rag_response のシグネチャを変更して渡す
            embedding_strategy_for_retrieval=embedding_strategy_for_retrieval
        )

        # Convert Document objects in sources to dicts for Pydantic model
        processed_sources = None
        if answer.get("sources"):
            processed_sources = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in answer["sources"]
            ]

        return ChatQueryResponse(
            answer=answer["answer"],
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
