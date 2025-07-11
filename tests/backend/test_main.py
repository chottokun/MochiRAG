import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Adjust import path to match how FastAPI apps are typically structured
# Assuming 'backend.main' can be imported if 'tests' is sibling to 'backend'
# and tests are run from project root.
# If running pytest from within 'tests' directory, this might need adjustment
# or PYTHONPATH manipulation.
from backend.main import app  # Direct import of the app instance

# Import models and other necessary components
from backend.models import User, Token, DataSourceMeta # DataSourceMeta を追加
from core.rag_chain import AVAILABLE_RAG_STRATEGIES # For testing strategies

client = TestClient(app)

# --- Test User Data ---
test_username = "testuser_main"
test_email = "testuser_main@example.com"
test_password = "testpassword123"

# --- Fixture for a test user and token ---
@pytest.fixture(scope="module")
def test_user_token():
    # Create user
    response = client.post(
        "/users/",
        json={"username": test_username, "email": test_email, "password": test_password},
    )
    # Handle case where user might already exist from previous test runs if DB is not cleared
    if response.status_code == 400 and "already exists" in response.json().get("detail", ""):
        # User already exists, try to log in to get token
        pass
    elif response.status_code != 200:
        pytest.fail(f"Failed to create test user: {response.status_code} - {response.text}")

    # Log in to get token
    login_response = client.post(
        "/token",
        data={"username": test_username, "password": test_password},
    )
    if login_response.status_code != 200:
        pytest.fail(f"Failed to log in test user: {login_response.status_code} - {login_response.text}")

    token_data = login_response.json()
    return {"access_token": token_data["access_token"], "token_type": token_data["token_type"]}

# --- Test Cases for /chat/query/ endpoint ---

# Mock the get_rag_response function from core.rag_chain
# This is important to isolate backend API tests from the RAG logic itself.
from langchain_core.documents import Document # Added import
import uuid # Added import

@patch("backend.main.get_rag_response")
def test_chat_query_successful_basic_strategy(mock_get_rag_response, test_user_token):
    mock_answer = "This is a mock RAG response."
    mock_source_docs = [
        Document(page_content="Source 1 content", metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content="Source 2 content", metadata={"source": "doc2.txt"})
    ]
    mock_get_rag_response.return_value = {
        "answer": mock_answer,
        "sources": mock_source_docs
    }

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "What is MochiRAG?",
        "rag_strategy": "basic" # Explicitly testing basic
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["answer"] == mock_answer
    assert response_data["strategy_used"] == "basic"
    assert "sources" in response_data

    expected_sources_api = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in mock_source_docs
    ]
    assert response_data["sources"] == expected_sources_api

    # Verify that get_rag_response was called with correct parameters
    mock_get_rag_response.assert_called_once()
    call_args_dict = mock_get_rag_response.call_args[1]
    assert call_args_dict["question"] == query_payload["question"]
    assert call_args_dict["rag_strategy"] == query_payload["rag_strategy"]
    assert "embedding_strategy_for_retrieval" in call_args_dict # これが渡されることを確認


@patch("backend.main.get_rag_response")
def test_chat_query_successful_deep_rag_strategy(mock_get_rag_response, test_user_token):
    if "deep_rag" not in AVAILABLE_RAG_STRATEGIES:
        pytest.skip("deep_rag strategy not in AVAILABLE_RAG_STRATEGIES, skipping test.")

    mock_answer_deep = "Mock response from DeepRAG."
    mock_source_docs_deep = [
        Document(page_content="Deep source 1", metadata={"source": "deep_doc1.md"}),
    ]
    mock_get_rag_response.return_value = {
        "answer": mock_answer_deep,
        "sources": mock_source_docs_deep
    }

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "Explain the MochiRAG system and its components in detail.",
        "rag_strategy": "deep_rag",
        "data_source_ids": ["sample_ds_id_for_deep_rag"] # テストのために仮のID
    }

    # _read_datasources_metaをモックして、特定のembedding_strategyを返すようにする
    # これは test_chat_query_uses_embedding_strategy_from_metadata と同様のモックが必要
    with patch("backend.main._read_datasources_meta") as mock_read_metas:
        # get_current_active_userもモックして一貫したユーザーIDを返す
        with patch("backend.main.auth.get_current_active_user") as mock_get_user:
            mock_user_instance = MagicMock(spec=User)
            mock_user_id_str = "deep_rag_test_user_id"
            mock_user_instance.user_id = mock_user_id_str # 文字列のUUIDとして
            mock_get_user.return_value = mock_user_instance

            mock_read_metas.return_value = {
                mock_user_id_str: [
                    MagicMock(data_source_id="sample_ds_id_for_deep_rag", embedding_strategy_used="default_embedding_for_deep_rag_test")
                ]
            }

            response = client.post("/chat/query/", headers=headers, json=query_payload)

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["answer"] == mock_answer_deep
            assert response_data["strategy_used"] == "deep_rag"
            assert "sources" in response_data

            expected_sources_api_deep = [
                {"page_content": doc.page_content, "metadata": doc.metadata} for doc in mock_source_docs_deep
            ]
            assert response_data["sources"] == expected_sources_api_deep

            mock_get_rag_response.assert_called_once()
            call_kwargs = mock_get_rag_response.call_args[1]
            assert call_kwargs["user_id"] == mock_user_id_str
            assert call_kwargs["question"] == query_payload["question"]
            assert call_kwargs["data_source_ids"] == query_payload["data_source_ids"]
            assert call_kwargs["rag_strategy"] == "deep_rag"
            assert call_kwargs["embedding_strategy_for_retrieval"] == "default_embedding_for_deep_rag_test"


def test_chat_query_invalid_strategy(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "This should fail.",
        "rag_strategy": "non_existent_strategy_123"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 400 # Bad Request due to invalid strategy
    response_data = response.json()
    assert "Invalid RAG strategy" in response_data["detail"]
    assert "non_existent_strategy_123" in response_data["detail"]

def test_chat_query_missing_question(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        # "question": "Missing question here", # Question is intentionally missing
        "rag_strategy": "basic"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation
    # FastAPI/Pydantic should catch this before our custom logic.

def test_chat_query_unauthorized():
    query_payload = {"question": "Test question", "rag_strategy": "basic"}
    response = client.post("/chat/query/", json=query_payload) # No auth header
    assert response.status_code == 401 # Unauthorized

@patch("backend.main.get_rag_response")
def test_chat_query_rag_function_exception(mock_get_rag_response, test_user_token):
    # Simulate an exception occurring within the get_rag_response function
    mock_get_rag_response.side_effect = Exception("Internal RAG error")

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    query_payload = {
        "question": "A question that causes an internal error.",
        "rag_strategy": "basic"
    }

    response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 500 # Internal Server Error
    response_data = response.json()
    # エラーメッセージの構造を確認し、それに合わせてアサーションを調整
    # "An error occurred processing your query with strategy '{selected_rag_strategy}': {str(e)}"
    assert f"An error occurred processing your query with strategy '{query_payload['rag_strategy']}'" in response_data["detail"]
    assert "Internal RAG error" in response_data["detail"] # Check if original error message is part of detail


# --- Dataset and Document Management Tests ---

def test_create_dataset_successful(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_payload = {"name": "My First Dataset", "description": "Test dataset description"}

    response = client.post("/users/me/datasets/", headers=headers, json=dataset_payload)

    assert response.status_code == 201 # Created
    data = response.json()
    assert data["name"] == dataset_payload["name"]
    assert data["description"] == dataset_payload["description"]
    assert "dataset_id" in data
    assert "user_id" in data # Should match the current user's ID (harder to assert directly without fetching user first)
    st.session_state.test_dataset_id = data["dataset_id"] # Save for later tests, if running sequentially in a class or needing state

def test_create_dataset_duplicate_name(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_payload = {"name": "My First Dataset", "description": "Another description"} # Same name as above

    # First creation (or ensure it exists from previous test if not isolated)
    # To make this test robust, we could try to create it here if it might not exist,
    # but for simplicity, assuming test_create_dataset_successful runs first or a dataset with this name exists.
    # client.post("/users/me/datasets/", headers=headers, json={"name": "My First Dataset", "description": "Initial"})

    response = client.post("/users/me/datasets/", headers=headers, json=dataset_payload)

    assert response.status_code == 400 # Bad Request
    assert "already exists" in response.json()["detail"]

def test_list_datasets(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    # Ensure at least one dataset is created (e.g., "My First Dataset")
    client.post("/users/me/datasets/", headers=headers, json={"name": "Dataset For Listing Test", "description": "List me"})


    response = client.get("/users/me/datasets/", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(ds["name"] == "My First Dataset" for ds in data)
    assert any(ds["name"] == "Dataset For Listing Test" for ds in data)

# Store dataset_id globally in the module for subsequent tests. This is a common pattern for test dependencies.
# However, pytest fixtures are generally preferred for managing dependencies and state across tests.
# For simplicity in this script, we'll use a module-level variable if needed, or pass IDs.
# A better way: create a dataset in a fixture if many tests depend on it.

@pytest.fixture(scope="module")
def created_dataset(test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_name = f"TestDataset_{uuid.uuid4().hex[:6]}"
    payload = {"name": dataset_name, "description": "A dataset created by a fixture"}
    response = client.post("/users/me/datasets/", headers=headers, json=payload)
    assert response.status_code == 201
    return response.json()


# --- Placeholder for Document Upload and Listing Tests (if relevant to current changes) ---
# These tests would typically reside here but are not directly affected by RAG strategy changes.
# For completeness, one might ensure they still work.

def test_upload_document_unsupported_type_to_dataset(test_user_token, created_dataset):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_id = created_dataset["dataset_id"]

    files = {'file': ('unsupported.xyz', b'some content', 'application/octet-stream')}
    # The new endpoint for dataset-specific upload
    response = client.post(f"/users/me/datasets/{dataset_id}/documents/upload/", headers=headers, files=files)

    assert response.status_code == 400
    # Detail message for deprecated endpoint has been updated in main.py
    assert "This endpoint is deprecated" in response.json()["detail"]


@patch("backend.main.load_and_split_document") # Mock the document processing
@patch("backend.main.vector_store_manager.add_documents")
def test_upload_document_to_dataset_successful(mock_vsm_add_docs, mock_load_split, test_user_token, created_dataset):
    mock_load_split.return_value = [MagicMock(spec=Document)] # Simulate one chunk
    mock_vsm_add_docs.return_value = 1 # Simulate one chunk added to VSM

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_id = created_dataset["dataset_id"]
    file_content = b"Content for a test file in a dataset."
    files = {'file': ('dataset_doc.txt', file_content, 'text/plain')}
    form_data = {
        "embedding_strategy": "default", # Assuming "default" is a valid strategy
        "chunking_strategy": "default",  # Assuming "default" is valid
    }

    response = client.post(
        f"/users/me/datasets/{dataset_id}/documents/upload/",
        headers=headers,
        files=files,
        data=form_data
    )

    assert response.status_code == 200
    data = response.json()
    assert data["original_filename"] == "dataset_doc.txt"
    assert data["dataset_id"] == dataset_id
    assert data["status"] == "processed"
    assert data["chunk_count"] == 1 # From mock_load_split and mock_vsm_add_docs
    assert data["embedding_strategy_used"] is not None # Default was used
    assert data["chunking_strategy_used"] is not None # Default was used

    mock_load_split.assert_called_once()
    # TODO: Assert call_args for load_and_split_document if specific params are important

    mock_vsm_add_docs.assert_called_once()
    vsm_call_kwargs = mock_vsm_add_docs.call_args[1]
    assert vsm_call_kwargs["dataset_id"] == dataset_id
    assert vsm_call_kwargs["embedding_strategy_name"] == "default" # Or actual default name
    assert vsm_call_kwargs["chunking_strategy_name"] == "default" # Or actual default name

def test_list_documents_in_dataset(test_user_token, created_dataset):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_id = created_dataset["dataset_id"]

    # First, upload a document to ensure the dataset is not empty
    with patch("backend.main.load_and_split_document", return_value=[MagicMock(spec=Document)]), \
         patch("backend.main.vector_store_manager.add_documents", return_value=1):
        client.post(
            f"/users/me/datasets/{dataset_id}/documents/upload/",
            headers=headers,
            files={'file': ('list_test_doc.txt', b"content", 'text/plain')},
            data={"embedding_strategy": "default", "chunking_strategy": "default"}
        )

    response = client.get(f"/users/me/datasets/{dataset_id}/documents/", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1 # Should have at least the document uploaded above
    assert any(doc["original_filename"] == "list_test_doc.txt" for doc in data)
    assert all(doc["dataset_id"] == dataset_id for doc in data)


@patch("backend.main.get_rag_response")
def test_chat_query_with_dataset_id(mock_get_rag_response, test_user_token, created_dataset):
    mock_answer = "Response from a specific dataset query."
    mock_get_rag_response.return_value = {"answer": mock_answer, "sources": []}

    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_id = created_dataset["dataset_id"]

    # Mock _read_datasources_meta to return some file for this dataset
    # This is to ensure the backend logic can find data_source_ids for the dataset
    mock_file_meta_in_dataset = DataSourceMeta(
        data_source_id=f"file_in_{dataset_id}.txt",
        dataset_id=uuid.UUID(dataset_id), # Ensure it's UUID
        original_filename=f"file_in_{dataset_id}.txt",
        status="processed",
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=1,
        embedding_strategy_used="default" # For retrieval strategy determination
    )

    # We need to mock what _read_datasources_meta() returns
    # It returns Dict[str, Dict[str, List[DataSourceMeta]]]
    # Get current user's ID from token for mocking (though it's a bit indirect)
    # A better approach would be to have a fixture for the current_user object.
    user_id_from_token = json.loads(base64.b64decode(test_user_token['access_token'].split('.')[1] + "==").decode())['user_id']

    mock_datasources_data = {
        user_id_from_token: {
            dataset_id: [mock_file_meta_in_dataset.model_dump(mode='json')] # Ensure it's JSON serializable for the mock
        }
    }

    # Patch _read_datasources_meta to return our controlled data
    # Also, ensure the mock can handle Pydantic model parsing if it happens after json.load
    # For simplicity, if _read_datasources_meta parses from dict to Pydantic, the mock should return dicts.
    # If it returns Pydantic, the mock should return Pydantic.
    # Based on current main.py, _read_datasources_meta returns Dict[str, Dict[str, List[DataSourceMeta]]] (Pydantic models)
    # So, the mock should return this structure.

    # Re-mocking the return value to be list of Pydantic models
    pydantic_mock_datasources_data = {
        user_id_from_token: {
            dataset_id: [mock_file_meta_in_dataset] # list of DataSourceMeta objects
        }
    }

    with patch("backend.main._read_datasources_meta", return_value=pydantic_mock_datasources_data):
        query_payload = {
            "question": "Query specifically for dataset_id",
            "dataset_ids": [dataset_id], # Pass dataset_id as a list
            "rag_strategy": "basic"
        }
        response = client.post("/chat/query/", headers=headers, json=query_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == mock_answer

    mock_get_rag_response.assert_called_once()
    call_args = mock_get_rag_response.call_args[1]
    assert call_args["question"] == query_payload["question"]
    # Backend should resolve dataset_ids to the list of data_source_ids within that dataset
    assert mock_file_meta_in_dataset.data_source_id in call_args["data_source_ids"]
    assert len(call_args["data_source_ids"]) == 1 # Only one file was in our mock dataset


# `/chat/query/` のテストで `embedding_strategy_for_retrieval` の検証を追加する
# そのためには、事前にDataSourceMetaに `embedding_strategy_used` を含むデータをモックする必要がある
from core.embedding_manager import embedding_manager # Added import

from unittest.mock import mock_open # Added import

# Skipを解除しました
@patch("backend.main.embedding_manager", autospec=True) # embedding_manager をモック
# @patch("backend.main._read_datasources_meta") # _read_datasources_meta のモックは解除
@patch("json.load") # json.load をモック
@patch("builtins.open", new_callable=mock_open) # open をモック
@patch("backend.main.get_rag_response")
def test_chat_query_uses_embedding_strategy_from_metadata(
    mock_get_rag_response, mock_open_file, mock_json_load, mock_embedding_manager, test_user_token # mock_read_metas を削除し、新しいモックを追加
):
    mock_embedding_manager.get_available_strategies.return_value = ["default_from_mock_manager"]
    mock_answer_meta = "Response based on specific embedding."
    # mock_get_rag_response.return_value は side_effect で処理するので削除

    # モックするメタデータ
    test_ds_id = "ds_with_specific_embedding"
    specific_embedding_strategy = "sentence_transformer_custom_test_model" # 仮の戦略名

    # auth.get_current_active_user のモック設定
    mock_user_instance = MagicMock(spec=User)
    user_id_for_meta_str = "123e4567-e89b-12d3-a456-426614174000"
    user_id_uuid = uuid.UUID(user_id_for_meta_str)
    mock_user_instance.user_id = user_id_uuid

    # _read_datasources_meta が読み込むであろうJSONデータをモックで設定
    ds_meta_dict = { # DataSourceMeta オブジェクトではなく、JSONからロードされる辞書を模倣
        "data_source_id": test_ds_id,
        "original_filename": "dummy.pdf",
        "status": "processed",
        "uploaded_at": "2023-01-01T00:00:00Z",
        "chunk_count": 10, # Optionalだが、テストのため設定
        "embedding_strategy_used": specific_embedding_strategy,
        # "additional_info": None, # Optional
        # "chunking_strategy_used": None, # Optional
        # "chunking_config_used": None, # Optional
    }
    mock_json_load.return_value = { # json.load が返すデータ
        user_id_for_meta_str: [ds_meta_dict]
    }

    # `open` が呼ばれたことを確認する程度 (実際のファイルパスは動的なので検証難しい)
    # mock_open_file.assert_called_with(DATASOURCES_META_PATH, "r", encoding="utf-8")

    with patch("backend.main.auth.get_current_active_user", return_value=mock_user_instance):
        headers = {"Authorization": f"Bearer {test_user_token['access_token']}"} # トークン自体は有効なものを使用
        query_payload = {
            "question": "Query for doc with specific embedding",
            "data_source_ids": [test_ds_id],
            "rag_strategy": "basic"
        }

        def get_rag_response_side_effect_checker(*args, **kwargs_of_get_rag_response):
            # この関数が get_rag_response の代わりとして呼ばれる
            # ここで backend/main.py が渡した embedding_strategy_for_retrieval を確認
            assert kwargs_of_get_rag_response.get("embedding_strategy_for_retrieval") == specific_embedding_strategy
            return { "answer": mock_answer_meta, "sources": [] }

        mock_get_rag_response.side_effect = get_rag_response_side_effect_checker

        response = client.post("/chat/query/", headers=headers, json=query_payload)

        assert response.status_code == 200
        mock_get_rag_response.assert_called_once()
        # call_kwargs のチェックは side_effect 内で行われたので、ここでは呼び出し回数の確認のみで十分


@patch("backend.main.vector_store_manager.delete_documents")
def test_delete_document_from_dataset(mock_vsm_delete_docs, test_user_token, created_dataset):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    dataset_id = created_dataset["dataset_id"]

    # 1. Upload a document first to have something to delete
    file_to_delete_name = f"to_delete_{uuid.uuid4().hex[:4]}.txt"
    with patch("backend.main.load_and_split_document", return_value=[MagicMock(spec=Document)]), \
         patch("backend.main.vector_store_manager.add_documents", return_value=1):
        upload_resp = client.post(
            f"/users/me/datasets/{dataset_id}/documents/upload/",
            headers=headers,
            files={'file': (file_to_delete_name, b"delete content", 'text/plain')},
            data={"embedding_strategy": "default", "chunking_strategy": "default"}
        )
        assert upload_resp.status_code == 200
        uploaded_file_meta = upload_resp.json()
        data_source_id_to_delete = uploaded_file_meta["data_source_id"]

    # 2. List documents to confirm it's there
    list_resp_before = client.get(f"/users/me/datasets/{dataset_id}/documents/", headers=headers)
    assert list_resp_before.status_code == 200
    assert any(doc["data_source_id"] == data_source_id_to_delete for doc in list_resp_before.json())

    # 3. Delete the document
    delete_resp = client.delete(
        f"/users/me/datasets/{dataset_id}/documents/{data_source_id_to_delete}/",
        headers=headers
    )
    assert delete_resp.status_code == 204 # No Content
    mock_vsm_delete_docs.assert_called_once_with(
        user_id=json.loads(base64.b64decode(test_user_token['access_token'].split('.')[1] + "==").decode())['user_id'], # Approximate user_id
        data_source_id=data_source_id_to_delete,
        dataset_id=dataset_id # Ensure dataset_id is passed to VSM delete if it uses it
    )

    # 4. List documents again to confirm it's gone from metadata
    list_resp_after = client.get(f"/users/me/datasets/{dataset_id}/documents/", headers=headers)
    assert list_resp_after.status_code == 200
    assert not any(doc["data_source_id"] == data_source_id_to_delete for doc in list_resp_after.json())


@patch("backend.main.vector_store_manager.delete_documents") # Mock VSM for file deletions within dataset
def test_delete_dataset_cascades(mock_vsm_delete_file_docs, test_user_token):
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}

    # 1. Create a new dataset specifically for this test
    dataset_name_for_delete_test = f"DatasetToDelete_{uuid.uuid4().hex[:6]}"
    create_ds_payload = {"name": dataset_name_for_delete_test, "description": "Test cascade delete"}
    create_ds_resp = client.post("/users/me/datasets/", headers=headers, json=create_ds_payload)
    assert create_ds_resp.status_code == 201
    dataset_to_delete_id = create_ds_resp.json()["dataset_id"]

    # 2. Upload a document to this dataset
    doc_name_in_deleted_ds = "doc_in_deleted_ds.txt"
    with patch("backend.main.load_and_split_document", return_value=[MagicMock(spec=Document)]), \
         patch("backend.main.vector_store_manager.add_documents", return_value=1): # This mock is for add_documents
        upload_resp = client.post(
            f"/users/me/datasets/{dataset_to_delete_id}/documents/upload/",
            headers=headers,
            files={'file': (doc_name_in_deleted_ds, b"content", 'text/plain')},
            data={"embedding_strategy": "default", "chunking_strategy": "default"}
        )
        assert upload_resp.status_code == 200
        uploaded_doc_meta = upload_resp.json()
        # data_source_id_of_doc_in_deleted_ds = uploaded_doc_meta["data_source_id"]


    # 3. Delete the dataset
    delete_ds_resp = client.delete(f"/users/me/datasets/{dataset_to_delete_id}/", headers=headers)
    assert delete_ds_resp.status_code == 204

    # Assert that VSM's delete_documents was called for the file within the dataset
    # This assumes delete_dataset correctly identifies files and calls VSM delete for each.
    # The number of calls depends on how many files were in the dataset. Here, 1 file.
    assert mock_vsm_delete_file_docs.call_count >= 1
    # We can be more specific if we capture the data_source_id of the uploaded doc
    # mock_vsm_delete_file_docs.assert_any_call(
    #     user_id=ANY, # Or capture user_id
    #     data_source_id=data_source_id_of_doc_in_deleted_ds,
    #     # dataset_id=dataset_to_delete_id # If VSM delete uses dataset_id
    # )

    # 4. Verify the dataset is gone from the list of datasets
    list_datasets_resp = client.get("/users/me/datasets/", headers=headers)
    assert list_datasets_resp.status_code == 200
    assert not any(ds["dataset_id"] == dataset_to_delete_id for ds in list_datasets_resp.json())

    # 5. Verify (indirectly) that file metadata for that dataset is also gone
    # This is harder to check directly without exposing an endpoint for all file metadata.
    # But if the dataset is gone, trying to list files for it should fail.
    list_files_deleted_ds_resp = client.get(f"/users/me/datasets/{dataset_to_delete_id}/documents/", headers=headers)
    assert list_files_deleted_ds_resp.status_code == 404 # Because the dataset itself is gone


# --- Example of a more involved test for document processing (if needed) ---
# @patch("backend.main.vector_store_manager.add_documents") # 修正: VSMのメソッドをモック
# @patch("backend.main.add_documents_to_vector_db")
# def test_upload_document_successful_txt(mock_add_to_db, mock_load_split, test_user_token):
#     mock_load_split.return_value = [MagicMock()] # Return a list with one dummy Document chunk
#     mock_add_to_db.return_value = None

#     headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
#     file_content = b"This is a test text file."
#     files = {'file': ('test.txt', file_content, 'text/plain')}

#     response = client.post("/documents/upload/", headers=headers, files=files)

#     assert response.status_code == 200
#     response_data = response.json()
#     assert response_data["original_filename"] == "test.txt"
#     assert response_data["status"] == "processed"
#     assert response_data["chunk_count"] == 1 # Based on mock_load_split

#     mock_load_split.assert_called_once()
#     # Path will be some temp path, so difficult to assert exactly without more mocking of Path/shutil
#     # Can assert that the file_type was 'txt'
#     assert mock_load_split.call_args[0][1] == 'txt'

#     mock_add_to_db.assert_called_once()
#     # Can assert user_id, data_source_id, and documents structure if needed.
#     assert mock_add_to_db.call_args[1]['user_id'] is not None # User ID is dynamic
#     assert mock_add_to_db.call_args[1]['data_source_id'].startswith("test.txt_")


if __name__ == "__main__":
    pytest.main()
