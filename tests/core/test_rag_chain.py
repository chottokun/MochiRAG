import pytest
from unittest.mock import patch, MagicMock
from core.rag_chain import get_rag_response, format_docs_with_sources, AVAILABLE_RAG_STRATEGIES
from langchain_core.documents import Document
from langchain_core.messages import AIMessage # Added import
from langchain_ollama import ChatOllama # Added import
from langchain_core.retrievers import BaseRetriever # Added import

# --- Tests for format_docs_with_sources ---

def test_format_docs_with_sources_empty():
    assert format_docs_with_sources([]) == "No context documents found."

def test_format_docs_with_sources_basic():
    docs = [
        Document(page_content="Content of doc1.", metadata={"original_filename": "doc1.txt", "data_source_id": "ds1"}),
        Document(page_content="Content of doc2.", metadata={"data_source_id": "ds2", "page": 1})
    ]
    result = format_docs_with_sources(docs)
    assert "Source (ID: ds1, Original: doc1.txt)" in result
    assert "Content of doc1." in result
    assert "Source (ID: ds2), Page: 2" in result # page is 0-indexed, so page 1 becomes Page: 2
    assert "Content of doc2." in result
    assert "---\n\n" in result # Separator

def test_format_docs_with_sources_minimal_metadata():
    docs = [Document(page_content="Minimal content.")]
    result = format_docs_with_sources(docs)
    assert "Unknown Source" in result
    assert "Minimal content." in result

# --- Tests for get_rag_response ---

@pytest.fixture
def mock_retriever_manager():
    with patch('core.rag_chain.retriever_manager', autospec=True) as mock_rm:
        yield mock_rm

@pytest.fixture
def mock_llm_manager():
    with patch('core.rag_chain.llm_manager', autospec=True) as mock_lm:
        # Mock the get_llm() method to return a MagicMock that can be configured per test
        mock_llm_instance = MagicMock(spec=ChatOllama) # spec を追加
        mock_lm.get_llm.return_value = mock_llm_instance
        yield mock_lm


@pytest.mark.parametrize("strategy", AVAILABLE_RAG_STRATEGIES)
def test_get_rag_response_returns_dict_with_answer_and_sources(mock_retriever_manager, mock_llm_manager, strategy):
    """
    Tests that get_rag_response returns a dictionary with 'answer' and 'sources' keys,
    and that 'sources' contains the documents from the retriever.
    """
    # Arrange
    user_id = "test_user"
    question = "What is RAG?"

    mock_retrieved_docs = [
        Document(page_content="RAG is Retrieval Augmented Generation.", metadata={"source": "doc1.txt"}),
        Document(page_content="It combines retrieval with generation.", metadata={"source": "doc2.pdf"})
    ]

    mock_retriever_instance = MagicMock(spec=BaseRetriever) # spec を追加
    mock_retriever_instance.invoke.return_value = mock_retrieved_docs # Retriever.invoke is called by LCEL
    mock_retriever_manager.get_retriever.return_value = mock_retriever_instance

    # Mock the LLM's response (StrOutputParser is the final step in the answer sub-chain)
    # The LLM instance itself is part of the chain, so its invoke method will be called.
    # We need to mock the behavior of the llm instance returned by llm_manager.get_llm()
    mock_llm_instance = mock_llm_manager.get_llm.return_value
    # LLM typically returns a message object, StrOutputParser extracts content.
    mock_llm_instance.invoke.return_value = AIMessage(content="This is a mock LLM answer.")

    # Act
    result = get_rag_response(
        user_id=user_id,
        question=question,
        rag_strategy=strategy, # type: ignore
        embedding_strategy_for_retrieval="test_embedding_strategy"
    )

    # Assert
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result

    assert isinstance(result["answer"], str)
    assert result["answer"] == "This is a mock LLM answer." # Check if LLM output is passed through

    assert isinstance(result["sources"], list)
    assert len(result["sources"]) == len(mock_retrieved_docs)
    for i, doc in enumerate(result["sources"]):
        assert isinstance(doc, Document)
        assert doc.page_content == mock_retrieved_docs[i].page_content
        assert doc.metadata == mock_retrieved_docs[i].metadata

    mock_retriever_manager.get_retriever.assert_called_once_with(
        name=strategy,
        user_id=user_id,
        embedding_strategy_name="test_embedding_strategy",
        data_source_ids=None # Default if not provided
    )
    # Check if the retriever was invoked with the question (this happens inside the LCEL chain)
    mock_retriever_instance.invoke.assert_called_once()
    retriever_call_args, _ = mock_retriever_instance.invoke.call_args
    assert retriever_call_args[0] == question

    # Check if the LLM was invoked (this also happens inside the LCEL chain)
    # The input to the LLM (via default_rag_prompt) will be a dict like {"context": "...", "question": "..."}
    mock_llm_instance.invoke.assert_called_once()
    # We could be more specific about the LLM input if needed, by inspecting mock_llm_instance.invoke.call_args

def test_get_rag_response_handles_retriever_failure(mock_retriever_manager, mock_llm_manager):
    """
    Tests that get_rag_response returns a fallback answer if the retriever setup fails.
    """
    mock_retriever_manager.get_retriever.side_effect = Exception("Retriever setup failed")

    result = get_rag_response(
        user_id="test_user",
        question="Any question",
        rag_strategy="basic",
        embedding_strategy_for_retrieval="test_embedding"
    )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "I'm sorry, but I encountered an error setting up the retrieval mechanism" in result["answer"]
    assert result["sources"] == []

def test_get_rag_response_handles_rag_chain_invocation_failure(mock_retriever_manager, mock_llm_manager):
    """
    Tests that get_rag_response returns a fallback answer if the RAG chain invocation fails.
    """
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.invoke.return_value = [Document(page_content="Test doc")]
    mock_retriever_manager.get_retriever.return_value = mock_retriever_instance

    # Mock the LLM to raise an exception when invoked
    mock_llm_instance = mock_llm_manager.get_llm.return_value
    mock_llm_instance.invoke.side_effect = Exception("LLM invocation failed")

    result = get_rag_response(
        user_id="test_user",
        question="Any question",
        rag_strategy="basic",
        embedding_strategy_for_retrieval="test_embedding"
    )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "I'm sorry, but I encountered an error while trying to generate a response" in result["answer"]
    assert result["sources"] == [] # Should still return empty sources list
