import unittest
from unittest.mock import patch, MagicMock
from core.ingestion_service import IngestionService, EmbeddingServiceError

class TestIngestionServiceV2(unittest.TestCase):
    def setUp(self):
        self.service = IngestionService()

    @patch("core.ingestion_service.vector_store_manager")
    @patch("core.ingestion_service.time.sleep") # Mock sleep to speed up tests
    def test_ingest_basic_retry_success(self, mock_sleep, mock_vsm):
        # Mock add_documents to fail twice and then succeed
        mock_vsm.add_documents.side_effect = [Exception("Error 1"), Exception("Error 2"), None]

        # Mock loader
        mock_loader = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "test content"
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader.load.return_value = [mock_doc]

        with patch.object(self.service, "_get_loader", return_value=mock_loader):
            self.service._ingest_basic("test.txt", "text/plain", 1, 1, 1)

        # Verify it was called 3 times
        self.assertEqual(mock_vsm.add_documents.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("core.ingestion_service.vector_store_manager")
    @patch("core.ingestion_service.time.sleep")
    def test_ingest_basic_retry_failure(self, mock_sleep, mock_vsm):
        # Mock add_documents to always fail
        mock_vsm.add_documents.side_effect = Exception("Persistent Error")

        # Mock loader
        mock_loader = MagicMock()
        mock_loader.load.return_value = [MagicMock(page_content="test", metadata={"source": "test.txt"})]

        with patch.object(self.service, "_get_loader", return_value=mock_loader):
            with self.assertRaises(EmbeddingServiceError):
                self.service._ingest_basic("test.txt", "text/plain", 1, 1, 1)

        # Verify it was called max_retries (3) times
        self.assertEqual(mock_vsm.add_documents.call_count, 3)

if __name__ == "__main__":
    unittest.main()
