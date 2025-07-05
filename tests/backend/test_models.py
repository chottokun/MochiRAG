import pytest
from backend.models import DataSourceMeta, ChatQueryRequest, ChatQueryResponse
from uuid import uuid4
from datetime import datetime
from pydantic import ValidationError

# DataSourceMeta のバリデーションテスト

def test_datasource_meta_required_fields():
    meta = DataSourceMeta(
        data_source_id="file123",
        original_filename="test.txt",
        status="uploaded",
        uploaded_at=datetime.now().isoformat()
    )
    assert meta.data_source_id == "file123"
    assert meta.status == "uploaded"
    assert meta.chunk_count is None
    assert meta.additional_info is None

    # chunk_count, additional_info の型チェック
    meta2 = DataSourceMeta(
        data_source_id="file456",
        original_filename="test2.txt",
        status="processed",
        uploaded_at=datetime.now().isoformat(),
        chunk_count=10,
        additional_info={"foo": "bar"}
    )
    assert meta2.chunk_count == 10
    assert meta2.additional_info["foo"] == "bar"

    # 必須フィールドが足りない場合はエラー
    with pytest.raises(ValidationError):
        DataSourceMeta()

# ChatQueryRequest のバリデーション

def test_chatqueryrequest_validation():
    req = ChatQueryRequest(question="What is RAG?", data_source_ids=["id1", "id2"])
    assert req.question == "What is RAG?"
    assert req.data_source_ids == ["id1", "id2"]

    # data_source_ids 省略可
    req2 = ChatQueryRequest(question="test")
    assert req2.data_source_ids is None

    # question は必須
    with pytest.raises(ValidationError):
        ChatQueryRequest()

# ChatQueryResponse のバリデーション

def test_chatqueryresponse_validation():
    resp = ChatQueryResponse(answer="This is the answer.")
    assert resp.answer == "This is the answer."

    # answer は必須
    with pytest.raises(ValidationError):
        ChatQueryResponse()
