import pytest
from unittest.mock import patch, MagicMock

from core.retriever_manager import HydeRetrieverStrategy
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0] * 768 for _ in texts]
    def embed_query(self, text):
        return [1.0] * 768

class MockVectorStore:
    """A mock that simulates the original vector store."""
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._client = MagicMock()
        self._collection = MagicMock()
        self._collection.name = "test_collection"

    @property
    def embeddings(self):
        return self._embeddings

@patch('core.retriever_manager.Chroma')
@patch('core.retriever_manager.HypotheticalDocumentEmbedder')
def test_hyde_retriever_strategy_fix_is_correct(MockHypotheticalDocumentEmbedder, MockChroma):
    """
    Tests that the fixed HydeRetrieverStrategy implementation correctly
    creates a new Chroma vector store with the hyde embeddings.
    """
    # 1. Setup mocks for dependencies
    mock_llm = MagicMock(spec=BaseLLM)
    mock_config = MagicMock()
    mock_config.parameters.get.return_value = 5
    mock_base_embeddings = FakeEmbeddings()
    mock_vector_store = MockVectorStore(embeddings=mock_base_embeddings)

    # 2. This is the mock object that the application code will receive when it
    #    instantiates HypotheticalDocumentEmbedder.
    mock_hyde_embedder_instance = MockHypotheticalDocumentEmbedder.return_value

    # 3. Patch the managers to return our mocks/doubles
    with patch('core.retriever_manager.config_manager.get_retriever_config', return_value=mock_config), \
         patch('core.retriever_manager.vector_store_manager.get_vector_store', return_value=mock_vector_store), \
         patch('core.retriever_manager.llm_manager.get_llm', return_value=mock_llm):

        strategy = HydeRetrieverStrategy()
        strategy.get_retriever(user_id=1, dataset_ids=[1])

    # 4. Assertions to verify the fix
    # Assert that a new Chroma instance was created.
    MockChroma.assert_called_once()

    # Get the arguments passed to the Chroma constructor.
    _, kwargs = MockChroma.call_args

    # Assert that the new Chroma instance was created with our mock hyde embedder.
    assert kwargs['embedding_function'] == mock_hyde_embedder_instance, \
        "The new Chroma vector store was not created with the hyde embeddings."
