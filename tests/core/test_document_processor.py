import pytest
from core.document_processor import load_and_split_document, SUPPORTED_FILE_TYPES
from pathlib import Path
import tempfile

# テキストファイル、Markdown、PDFのロード・分割テスト

def test_load_and_split_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("A" * 1500)
    docs = load_and_split_document(str(file_path), "txt", chunk_size=1000, chunk_overlap=200)
    assert len(docs) == 2
    assert all(hasattr(doc, "page_content") for doc in docs)

def test_load_and_split_md(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\nContent" * 100)
    docs = load_and_split_document(str(file_path), "md", chunk_size=500, chunk_overlap=100)
    assert len(docs) >= 1
    assert all(hasattr(doc, "page_content") for doc in docs)

def test_load_and_split_pdf(tmp_path):
    # PDFLoaderのテストは本物のPDFが必要なので、ここではスキップ
    pytest.skip("PDFの自動テストはサンプルPDFが必要なためスキップ")

def test_load_and_split_unsupported(tmp_path):
    file_path = tmp_path / "sample.xyz"
    file_path.write_text("dummy")
    with pytest.raises(ValueError):
        load_and_split_document(str(file_path), "xyz")

def test_load_and_split_file_not_found():
    with pytest.raises(RuntimeError):
        load_and_split_document("/not/exist.txt", "txt")
