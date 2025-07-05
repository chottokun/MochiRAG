import logging
from typing import List, Optional, Dict, Any, Literal

# Langchain component imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever as LangchainParentDocumentRetriever


# Project-specific imports
try:
    from core.vector_store import query_vector_db, add_documents_to_vector_db, vector_db_client, embedding_function
    from core.document_processor import text_splitter # Assuming text_splitter is exposed
except ImportError:
    # Fallback for running as a script directly from core directory or if PYTHONPATH isn't set
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.vector_store import query_vector_db, add_documents_to_vector_db, vector_db_client, embedding_function
    from core.document_processor import text_splitter


# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
try:
    llm = ChatOllama(model="gemma3:4b-it-qat", temperature=0)
    logger.info("ChatOllama LLM initialized with model 'gemma3:4b-it-qat'.")
except Exception as e:
    logger.error(f"Failed to initialize ChatOllama: {e}. Ensure Ollama is running and the 'gemma3:4b-it-qat' model is pulled.")
    raise

# --- RAG Strategy Types ---
RAG_STRATEGY_TYPE = Literal[
    "basic",
    "parent_document",
    "multi_query",
    "contextual_compression"
]
AVAILABLE_RAG_STRATEGIES = list(RAG_STRATEGY_TYPE.__args__)


# --- Prompt Templates ---
DEFAULT_RAG_PROMPT_TEMPLATE_STR = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Try to keep the answer concise and informative.
When you use information from the context, cite the source using the metadata (e.g., "According to [document_name from metadata], ...").
The 'document_name' can be found in the 'original_filename' or 'data_source_id' fields of the source metadata.

Context:
{context}

Question: {question}

Answer:
"""
default_rag_prompt = PromptTemplate(template=DEFAULT_RAG_PROMPT_TEMPLATE_STR, input_variables=["context", "question"])

# For MultiQueryRetriever
QUERY_GEN_PROMPT_TEMPLATE_STR = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines.
Original question: {question}"""
query_gen_prompt = PromptTemplate(template=QUERY_GEN_PROMPT_TEMPLATE_STR, input_variables=["question"])


# --- Document Formatting ---
def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Formats a list of documents into a single string for RAG context,
    including source information from metadata.
    """
    if not docs:
        return "No context documents found."

    formatted_docs = []
    for i, doc in enumerate(docs):
        source_info = "Unknown Source"
        if doc.metadata:
            original_filename = doc.metadata.get("original_filename")
            data_source_id = doc.metadata.get("data_source_id")
            if original_filename and data_source_id:
                source_info = f"Source (ID: {data_source_id}, Original: {original_filename})"
            elif data_source_id:
                source_info = f"Source (ID: {data_source_id})"
            elif original_filename:
                 source_info = f"Source (Original: {original_filename})"
            # Add page number if available
            page_num = doc.metadata.get("page")
            if page_num is not None:
                source_info += f", Page: {page_num + 1}" # PyPDFLoader pages are 0-indexed

        formatted_docs.append(f"{source_info}\nContent: {doc.page_content}")

    return "\n\n---\n\n".join(formatted_docs)


# --- Retriever Implementations ---

def get_base_retriever(user_id: str, data_source_ids: Optional[List[str]] = None, n_results: int = 3):
    """Returns a basic retriever from the vector store."""
    search_kwargs = {"k": n_results}
    if data_source_ids:
        # Build the filter for ChromaDB
        # Assuming user_id is always present and data_source_ids might be a list or None
        conditions = [{"user_id": user_id}]
        if data_source_ids:
            if len(data_source_ids) == 1:
                conditions.append({"data_source_id": data_source_ids[0]})
            else:
                conditions.append({"data_source_id": {"$in": data_source_ids}})

        final_filter: Dict[str, Any]
        if len(conditions) > 1:
            final_filter = {"$and": conditions}
        else: # Only user_id
            final_filter = conditions[0]
        search_kwargs["filter"] = final_filter
    else: # Only user_id based retrieval
        search_kwargs["filter"] = {"user_id": user_id}

    return vector_db_client.as_retriever(search_kwargs=search_kwargs)

def get_parent_document_retriever(user_id: str, data_source_ids: Optional[List[str]] = None, n_results: int = 3):
    """
    Returns a ParentDocumentRetriever.
    It retrieves smaller chunks for similarity search and then looks up the parent documents.
    """
    # The store for parent documents
    docstore = InMemoryStore() # Could be replaced with a persistent store if needed

    # Create the ParentDocumentRetriever
    # text_splitter needs to be the one used for chunking during ingestion, or compatible
    # For MochiRAG, child_splitter is the RecursiveCharacterTextSplitter from document_processor
    # parent_splitter can be configured to split by e.g. paragraphs or keep larger documents.
    # For simplicity, we'll assume the default RecursiveCharacterTextSplitter is fine for child splitting.
    # The parent documents are implicitly the full documents loaded before splitting.
    # We need to ensure that when documents are added via `add_documents_to_vector_db`,
    # they are also added to the `docstore` for this retriever to work.
    # This implies `add_documents_to_vector_db` might need modification or a parallel mechanism.
    # For now, this setup assumes `vector_db_client` contains the child chunks.

    # This retriever type is more complex to set up correctly without modifying ingestion.
    # A simpler approach for "parent document" might be to retrieve more context around a found chunk.
    # However, the official ParentDocumentRetriever is designed for specific ingestion patterns.
    # Let's try to make it work by assuming vector_db_client is the child chunk store
    # and we'd need a way to populate docstore with parent docs.
    # This is a placeholder for a more robust implementation that integrates with ingestion.
    # For now, it will likely behave like a basic retriever if docstore is empty or not correctly populated.

    base_retriever = get_base_retriever(user_id, data_source_ids, n_results)

    # To make ParentDocumentRetriever work, you typically ingest documents through it.
    # Since we are not doing that here, this will not function as intended without further changes
    # to how documents are stored and indexed.
    # For this exercise, we will return the base_retriever and note this limitation.
    logger.warning("ParentDocumentRetriever requires specific ingestion setup. Falling back to basic retriever for now.")
    return base_retriever # Placeholder - needs proper docstore integration.

    # Correct setup would involve:
    # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000) # Example
    # child_splitter = RecursiveCharacterTextSplitter(chunk_size=400) # Example, should match ingestion
    # big_chunks_retriever = LangchainParentDocumentRetriever(
    #     vectorstore=vector_db_client, # This should be the store of child chunks
    #     docstore=docstore,            # This should be the store of parent documents
    #     child_splitter=child_splitter,
    #     # parent_splitter=parent_splitter # Optional, if you want to split parents further
    # )
    # This retriever needs `add_documents` to be called on it with the original full documents.
    # return big_chunks_retriever


def get_multi_query_retriever(user_id: str, data_source_ids: Optional[List[str]] = None, n_results: int = 3):
    """Returns a MultiQueryRetriever."""
    base_retriever = get_base_retriever(user_id, data_source_ids, n_results)
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, prompt=query_gen_prompt
    )

def get_contextual_compression_retriever(user_id: str, data_source_ids: Optional[List[str]] = None, n_results: int = 5):
    """Returns a ContextualCompressionRetriever."""
    base_retriever = get_base_retriever(user_id, data_source_ids, n_results) # Retrieve more initially
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )


# --- Main RAG Function ---

def get_rag_response(
    user_id: str,
    question: str,
    data_source_ids: Optional[List[str]] = None,
    rag_strategy: RAG_STRATEGY_TYPE = "basic"
) -> str:
    """
    Retrieves relevant documents using the specified RAG strategy and generates a response.
    """
    logger.info(f"Getting RAG response for user '{user_id}', strategy: '{rag_strategy}', question: '{question[:50]}...'")

    # 1. Select and initialize retriever based on strategy
    if rag_strategy == "basic":
        retriever = get_base_retriever(user_id, data_source_ids, n_results=3)
    elif rag_strategy == "parent_document":
        # Note: This currently falls back to basic due to setup complexity.
        retriever = get_parent_document_retriever(user_id, data_source_ids, n_results=3)
    elif rag_strategy == "multi_query":
        retriever = get_multi_query_retriever(user_id, data_source_ids, n_results=3)
    elif rag_strategy == "contextual_compression":
        retriever = get_contextual_compression_retriever(user_id, data_source_ids, n_results=5) # Retrieve more for compressor
    else:
        logger.warning(f"Unknown RAG strategy: '{rag_strategy}'. Defaulting to 'basic'.")
        retriever = get_base_retriever(user_id, data_source_ids, n_results=3)

    # 2. Define the RAG chain using LCEL
    rag_chain = (
        RunnableParallel(
            context=(RunnableLambda(lambda x: x["question"]) | retriever | format_docs_with_sources),
            question=RunnablePassthrough() # Pass the original question through
        )
        | RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]["question"]}) # Ensure correct dict keys for prompt
        | default_rag_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Invoke the chain
    try:
        # The input to the chain should be a dictionary containing the question
        response_text = rag_chain.invoke({"question": question})
        logger.info(f"Generated RAG response: {response_text[:100]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error invoking RAG chain with strategy '{rag_strategy}': {e}", exc_info=True)
        return f"I'm sorry, but I encountered an error while trying to generate a response using the '{rag_strategy}' strategy."


if __name__ == "__main__":
    logger.info("--- Running RAG Chain Test with Strategies ---")

    test_user_id = "rag_test_user_strat"
    test_ds_id_strat = "rag_test_source_strat"

    logger.info("Adding a dummy document for RAG strategy tests...")
    dummy_docs_for_strat_test = [
        Document(
            page_content="MochiRAG is a flexible system. It supports various RAG strategies like basic retrieval, parent document retrieval, multi-query, and contextual compression. This allows users to test different approaches for question answering.",
            metadata={"original_filename": "mochirag_strategies.txt", "page": 1, "data_source_id": test_ds_id_strat, "user_id": test_user_id}
        ),
        Document(
            page_content="The core idea of MochiRAG is to augment Large Language Models with external knowledge retrieved from user-provided documents. This improves factual consistency and reduces hallucinations.",
            metadata={"original_filename": "mochirag_core_idea.md", "page": 1, "data_source_id": test_ds_id_strat, "user_id": test_user_id}
        ),
        Document(
            page_content="When using multi-query strategy, the system generates several variations of the original question to broaden the search scope. Contextual compression aims to extract only the relevant parts of retrieved documents.",
            metadata={"original_filename": "mochirag_advanced_strategies.txt", "page": 1, "data_source_id": test_ds_id_strat, "user_id": test_user_id}
        )
    ]
    try:
        # We need to add these directly to the vector store for the retrievers to find them.
        # The `add_documents_to_vector_db` function in vector_store.py handles metadata correctly.
        add_documents_to_vector_db(test_user_id, test_ds_id_strat, dummy_docs_for_strat_test)
        logger.info("Dummy documents added for RAG strategy test.")
    except Exception as e:
        logger.error(f"Could not add dummy documents for RAG strategy test: {e}. Tests might fail.", exc_info=True)


    questions = [
        "What RAG strategies does MochiRAG support?",
        "How does multi-query strategy work in MochiRAG?",
        "What is the main purpose of MochiRAG?"
    ]

    for strategy in AVAILABLE_RAG_STRATEGIES:
        logger.info(f"\n--- Testing Strategy: {strategy.upper()} ---")
        for q_idx, question_text in enumerate(questions):
            logger.info(f"Test Case {q_idx+1} ({strategy}): Question: '{question_text}' for user '{test_user_id}' using source '{test_ds_id_strat}'")
            try:
                response = get_rag_response(
                    user_id=test_user_id,
                    question=question_text,
                    data_source_ids=[test_ds_id_strat],
                    rag_strategy=strategy
                )
                print(f"\nResponse for Q{q_idx+1} ({strategy}):\n{response}")
            except Exception as e:
                logger.error(f"Error during RAG Test Case {q_idx+1} ({strategy}): {e}", exc_info=True)
                print(f"\nError getting response for Q{q_idx+1} ({strategy}): {e}")

    # Clean up dummy documents
    logger.info(f"\nCleaning up dummy documents for user '{test_user_id}' and source '{test_ds_id_strat}'...")
    try:
        from core.vector_store import delete_documents_by_metadata
        # Deleting based on the specific user_id and data_source_id used for this test run
        delete_documents_by_metadata({"user_id": test_user_id, "data_source_id": test_ds_id_strat})
        logger.info("Dummy documents for strategy test cleaned up.")
    except Exception as e:
        logger.error(f"Error cleaning up dummy strategy test documents: {e}", exc_info=True)

    logger.info("--- RAG Chain Strategy Test Finished ---")
