import unittest
from unittest.mock import patch, MagicMock, mock_open, call

# Import the class and instances we need to test and mock
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import BaseLLM
from langchain.embeddings import HypotheticalDocumentEmbedder
from core.retriever_manager import BasicRetrieverStrategy, HydeRetrieverStrategy
from core.config_manager import config_manager
from core.vector_store_manager import vector_store_manager
from core.llm_manager import llm_manager

class TestBasicRetrieverStrategy(unittest.TestCase):
    # --- All tests for BasicRetrieverStrategy are here ---
    def setUp(self):
        self.strategy = BasicRetrieverStrategy()
        self.user_id = 123
        self.mock_retriever = MagicMock(spec=BaseRetriever)

    @patch.object(vector_store_manager, 'get_vector_store')
    @patch.object(config_manager, 'get_retriever_config')
    def test_personal_and_shared_ensemble(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_get_vector_store.return_value.as_retriever.return_value = self.mock_retriever
        dataset_ids = [1, -1]
        shared_dbs_json = '[{"id": -1, "collection_name": "shared_coll_1"}]'
        with patch("builtins.open", mock_open(read_data=shared_dbs_json)):
            retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertIsInstance(retriever, EnsembleRetriever)
        self.assertEqual(len(retriever.retrievers), 2)

    @patch.object(vector_store_manager, 'get_vector_store')
    @patch.object(config_manager, 'get_retriever_config')
    def test_multiple_shared_collections_ensemble(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_get_vector_store.return_value.as_retriever.return_value = self.mock_retriever
        dataset_ids = [-1, -2]
        shared_dbs_json = '''
        [
            {"id": -1, "collection_name": "shared_coll_1"},
            {"id": -2, "collection_name": "shared_coll_2"}
        ]
        '''
        with patch("builtins.open", mock_open(read_data=shared_dbs_json)):
            retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertIsInstance(retriever, EnsembleRetriever)
        self.assertEqual(len(retriever.retrievers), 2)

    @patch.object(vector_store_manager, 'get_vector_store')
    @patch.object(config_manager, 'get_retriever_config')
    def test_personal_only_single_retriever(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_as_retriever = mock_get_vector_store.return_value.as_retriever
        mock_as_retriever.return_value = self.mock_retriever
        dataset_ids = [1, 2, 3]
        retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertNotIsInstance(retriever, EnsembleRetriever)
        self.assertEqual(retriever, self.mock_retriever)

    @patch.object(vector_store_manager, 'get_vector_store')
    @patch.object(config_manager, 'get_retriever_config')
    def test_no_dataset_ids_returns_default_user_retriever(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_as_retriever = mock_get_vector_store.return_value.as_retriever
        mock_as_retriever.return_value = self.mock_retriever
        retriever = self.strategy.get_retriever(self.user_id, None)
        self.assertIsNotNone(retriever)
        mock_get_vector_store.assert_called_once_with(f"user_{self.user_id}")

    @patch('builtins.print')
    @patch.object(vector_store_manager, 'get_vector_store')
    @patch.object(config_manager, 'get_retriever_config')
    def test_invalid_shared_id_is_handled_gracefully(self, mock_get_config, mock_get_vector_store, mock_print):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_get_vector_store.return_value.as_retriever.return_value = self.mock_retriever
        dataset_ids = [-1, -99] # -99 is invalid
        shared_dbs_json = '[{"id": -1, "collection_name": "shared_coll_1"}]'
        with patch("builtins.open", mock_open(read_data=shared_dbs_json)):
            self.strategy.get_retriever(self.user_id, dataset_ids)
        mock_print.assert_called_once_with("Warning: The following shared dataset IDs were not found in shared_dbs.json: [-99]")

    @patch.object(vector_store_manager, 'get_vector_store')
    @patch.object(config_manager, 'get_retriever_config')
    def test_empty_retriever_for_no_valid_ids(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_as_retriever = mock_get_vector_store.return_value.as_retriever
        dataset_ids = [-99]
        shared_dbs_json = '[{"id": -1, "collection_name": "shared_coll_1"}]'
        with patch("builtins.open", mock_open(read_data=shared_dbs_json)):
            self.strategy.get_retriever(self.user_id, dataset_ids)
        mock_as_retriever.assert_called_once_with(search_kwargs={'k': 0})

@patch('core.retriever_manager.Chroma')
@patch('core.retriever_manager.HypotheticalDocumentEmbedder')
@patch.object(llm_manager, 'get_llm')
@patch.object(vector_store_manager, 'get_vector_store')
@patch.object(config_manager, 'get_retriever_config')
class TestHydeRetrieverStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = HydeRetrieverStrategy()
        self.user_id = 123

    def configure_mocks(self, mock_get_config, mock_get_vector_store, mock_get_llm, MockHydeEmbedder, MockChroma):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_get_llm.return_value = MagicMock(spec=BaseLLM)

        mock_vector_store = MagicMock()
        mock_vector_store.embeddings = MagicMock()
        mock_vector_store._client = MagicMock()
        mock_vector_store._collection.name = "mock_collection"
        mock_get_vector_store.return_value = mock_vector_store

        MockHydeEmbedder.return_value = MagicMock()
        MockChroma.return_value.as_retriever.return_value = MagicMock(spec=BaseRetriever)

    def test_hyde_personal_and_shared_ensemble(self, *mocks):
        """Tests that Hyde strategy correctly creates an EnsembleRetriever."""
        self.configure_mocks(*mocks)
        dataset_ids = [1, -1]
        shared_dbs_json = '[{"id": -1, "collection_name": "shared_coll_1"}]'

        with patch("builtins.open", mock_open(read_data=shared_dbs_json)):
            retriever = self.strategy.get_retriever(self.user_id, dataset_ids)

        self.assertIsInstance(retriever, EnsembleRetriever)
        self.assertEqual(len(retriever.retrievers), 2)

    def test_hyde_personal_only(self, *mocks):
        """Tests that Hyde strategy returns a single retriever for personal only."""
        self.configure_mocks(*mocks)
        dataset_ids = [1, 2]

        retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertNotIsInstance(retriever, EnsembleRetriever)
        self.assertIsInstance(retriever, BaseRetriever)

    def test_hyde_no_ids(self, *mocks):
        """Tests that Hyde strategy returns a single retriever for no dataset ids."""
        self.configure_mocks(*mocks)
        retriever = self.strategy.get_retriever(self.user_id, None)
        self.assertNotIsInstance(retriever, EnsembleRetriever)
        self.assertIsInstance(retriever, BaseRetriever)

    @patch('builtins.print')
    def test_hyde_invalid_id(self, mock_print, *mocks):
        """Tests that Hyde strategy handles invalid ids gracefully."""
        self.configure_mocks(*mocks)
        dataset_ids = [-99]
        shared_dbs_json = '[{"id": -1, "collection_name": "shared_coll_1"}]'

        with patch("builtins.open", mock_open(read_data=shared_dbs_json)):
            self.strategy.get_retriever(self.user_id, dataset_ids)

        mock_print.assert_called_once_with("Warning: The following shared dataset IDs were not found in shared_dbs.json: [-99]")

if __name__ == '__main__':
    unittest.main()
