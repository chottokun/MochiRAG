import unittest
from unittest.mock import patch, MagicMock
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

# Modules to test
from core.retriever_manager import ACERetrieverStrategy, _BaseRetrieverAdapter
from core.context_evolution_service import ContextEvolutionService

# Mock data
mock_evolved_context = MagicMock()
mock_evolved_context.content = "This is an evolved insight."
mock_evolved_context.id = 1
mock_evolved_context.effectiveness_score = 10

# A mock retriever class that conforms to BaseRetriever for Pydantic validation
class MockBaseRetriever(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs

class TestACERetrieverStrategy(unittest.TestCase):

    @patch('core.retriever_manager.BasicRetrieverStrategy')
    @patch('core.retriever_manager.llm_manager')
    @patch('core.retriever_manager.crud')
    def test_ace_retriever_combines_docs(self, mock_crud, mock_llm_manager, mock_basic_strategy):
        # --- Setup ---
        user_id = 123
        question = "What is MochiRAG?"

        # Use the compliant mock retriever
        mock_base_retriever = MockBaseRetriever(
            docs=[
                Document(page_content="Standard document 1"),
                Document(page_content="Standard document 2")
            ]
        )
        mock_basic_strategy.return_value.get_retriever.return_value = mock_base_retriever

        # Mock the LLM chain to return a fixed topic
        mock_topic_chain = MagicMock(spec=Runnable)
        mock_topic_chain.invoke.return_value = "MochiRAG"

        mock_llm = MagicMock()
        mock_llm_manager.get_llm.return_value = mock_llm

        # Mock the CRUD function to return our mock evolved context
        mock_crud.get_evolved_contexts_by_topic.return_value = [mock_evolved_context]

        # --- Execution ---
        ace_strategy = ACERetrieverStrategy()
        ace_retriever = ace_strategy.get_retriever(user_id=user_id, dataset_ids=[1])
        # Replace the chain with our mock *after* initialization
        ace_retriever.topic_gen_chain = mock_topic_chain

        result_docs = ace_retriever.get_relevant_documents(question)

        # --- Assertions ---
        # 1. Verify that the topic generation chain was called
        mock_topic_chain.invoke.assert_called_with({"question": question}, config=unittest.mock.ANY)

        # 2. Verify that the CRUD function was called with the correct topic
        mock_crud.get_evolved_contexts_by_topic.assert_called_once_with(
            unittest.mock.ANY, # db session
            owner_id=user_id,
            topic="MochiRAG"
        )

        # 3. Verify that the final list of documents is correctly combined
        self.assertEqual(len(result_docs), 3)
        self.assertEqual(result_docs[0].page_content, "This is an evolved insight.")
        self.assertEqual(result_docs[0].metadata['source'], "evolved_context")
        self.assertEqual(result_docs[1].page_content, "Standard document 1")


class TestContextEvolutionService(unittest.TestCase):

    @patch('core.context_evolution_service.crud')
    @patch('core.context_evolution_service.llm_manager')
    def test_evolve_context_from_interaction(self, mock_llm_manager, mock_crud):
        # --- Setup ---
        user_id = 123
        question = "How does the ParentDocumentRetriever work?"
        answer = "It splits documents into parent and child chunks."

        # Mock the LLM and its chains to return actual strings, not mocks
        mock_llm = MagicMock()
        mock_llm_manager.get_llm.return_value = mock_llm

        # --- Execution ---
        service = ContextEvolutionService()

        # We now mock the chains directly to control their string output
        mock_topic_chain = MagicMock(spec=Runnable)
        mock_topic_chain.invoke.return_value = "ParentDocumentRetriever"
        service.topic_gen_chain = mock_topic_chain

        mock_evolution_chain = MagicMock(spec=Runnable)
        mock_evolution_chain.invoke.return_value = "The key insight is that ParentDocumentRetriever uses small chunks for searching and large chunks for context."
        service.evolution_chain = mock_evolution_chain

        service.evolve_context_from_interaction(user_id, question, answer)

        # --- Assertions ---
        # 1. Verify that the CRUD function was called with the generated content
        mock_crud.create_evolved_context.assert_called_once_with(
            db=unittest.mock.ANY,
            owner_id=user_id,
            content="The key insight is that ParentDocumentRetriever uses small chunks for searching and large chunks for context.",
            topic="ParentDocumentRetriever"
        )

if __name__ == '__main__':
    unittest.main()
