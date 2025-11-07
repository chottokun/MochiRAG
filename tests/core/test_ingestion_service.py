import pytest
from unittest.mock import patch, MagicMock
from core.ingestion_service import IngestionService

# We need to patch the loader classes at the source where they are imported
@patch('core.ingestion_service.DoclingLoader')
@patch('core.ingestion_service.TextLoader')
@patch('core.ingestion_service.UnstructuredMarkdownLoader')
def test_get_loader_pdf(mock_md_loader, mock_text_loader, mock_docling_loader):
    """
    Tests if _get_loader returns an instance of DoclingLoader for PDF files.
    """
    # Arrange
    ingestion_service = IngestionService()
    test_file_path = "/path/to/dummy.pdf"
    test_file_type = "application/pdf"

    # Act
    loader = ingestion_service._get_loader(test_file_path, test_file_type)

    # Assert
    mock_docling_loader.assert_called_once_with(test_file_path, chunking_strategy="MARKDOWN")
    assert loader is not None
    # Check if the returned object is the instance created by the mock
    assert loader == mock_docling_loader.return_value
    mock_text_loader.assert_not_called()
    mock_md_loader.assert_not_called()

def test_get_loader_txt():
    """
    Tests if _get_loader returns an instance of TextLoader for plain text files.
    """
    # This test is more about ensuring the routing logic works, so we can keep it simple
    ingestion_service = IngestionService()
    loader = ingestion_service._get_loader("/path/to/dummy.txt", "text/plain")
    assert loader.__class__.__name__ == "TextLoader"

def test_unsupported_file_type():
    """
    Tests if _get_loader returns None for an unsupported file type.
    """
    ingestion_service = IngestionService()
    loader = ingestion_service._get_loader("/path/to/dummy.xyz", "application/octet-stream")
    assert loader is None
