import unittest
from unittest.mock import patch, MagicMock
from core.retriever_manager import RetrieverManager, BasicRetrieverStrategy

class TestRetrieverManagerV2(unittest.TestCase):
    def test_get_retriever_success(self):
        rm = RetrieverManager()
        # Mock strategy to return a dummy retriever
        mock_retriever = MagicMock()
        rm.strategies["basic"] = MagicMock()
        rm.strategies["basic"].get_retriever.return_value = mock_retriever

        retriever = rm.get_retriever("basic", user_id=1)
        self.assertEqual(retriever, mock_retriever)

    def test_get_retriever_unknown_strategy(self):
        rm = RetrieverManager()
        with self.assertRaises(ValueError):
            rm.get_retriever("non_existent_strategy", user_id=1)

    @patch("core.retriever_manager.vector_store_manager")
    def test_basic_retriever_strategy_fallback(self, mock_vsm):
        # Test when no dataset_ids are provided, it fallbacks to user_id filter
        strategy = BasicRetrieverStrategy()
        mock_vs = MagicMock()
        mock_vsm.get_vector_store.return_value = mock_vs

        strategy.get_retriever(user_id=1)

        mock_vs.as_retriever.assert_called_once()
        # Verify it uses the user_id filter
        args, kwargs = mock_vs.as_retriever.call_args
        self.assertEqual(kwargs["search_kwargs"]["filter"], {"user_id": 1})

if __name__ == "__main__":
    unittest.main()
