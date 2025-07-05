import pytest
from core.rag_chain import format_docs_with_sources
from langchain_core.documents import Document

def test_format_docs_with_sources_empty():
    assert format_docs_with_sources([]) == "No context documents found."

def test_format_docs_with_sources_basic():
    docs = [
        Document(page_content="answer1", metadata={"original_filename": "doc1.txt"}),
        Document(page_content="answer2", metadata={"data_source_id": "src2"})
    ]
    result = format_docs_with_sources(docs)
    assert "doc1.txt" in result or "src2" in result
    assert "answer1" in result and "answer2" in result
