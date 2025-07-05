import pytest
from core.vector_store import add_documents_to_vector_db
from langchain_core.documents import Document

# vector_storeのadd_documents_to_vector_dbの基本動作テスト

def test_add_documents_to_vector_db(monkeypatch):
    user_id = "user1"
    data_source_id = "source1"
    docs = [Document(page_content="test content", metadata={})]

    # vector_db_client.add_documentsが呼ばれるかをモック
    called = {}
    def fake_add_documents(documents):
        called["ok"] = True
        assert len(documents) == 1
        assert documents[0].metadata["user_id"] == user_id
        assert documents[0].metadata["data_source_id"] == data_source_id
    monkeypatch.setattr("core.vector_store.vector_db_client.add_documents", fake_add_documents)

    add_documents_to_vector_db(user_id, data_source_id, docs)
    assert called["ok"]

# 空リストの場合は何もしない

def test_add_documents_to_vector_db_empty(monkeypatch):
    called = {}
    def fake_add_documents(documents):
        called["fail"] = True
    monkeypatch.setattr("core.vector_store.vector_db_client.add_documents", fake_add_documents)
    add_documents_to_vector_db("u", "d", [])
    assert "fail" not in called
