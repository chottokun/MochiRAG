import logging
from typing import List, Optional, Dict, Any

# Langchain component imports
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Project-specific imports
try:
    from core.vector_store import query_vector_db, add_documents_to_vector_db # add_documents for testing block
except ImportError:
    # Fallback for running as a script directly from core directory or if PYTHONPATH isn't set
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.vector_store import query_vector_db, add_documents_to_vector_db


# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
try:
    llm = ChatOllama(model="llama3", temperature=0) # Using temperature=0 for more deterministic output
    logger.info("ChatOllama LLM initialized with model 'llama3'.")
except Exception as e:
    logger.error(f"Failed to initialize ChatOllama: {e}. Ensure Ollama is running and the 'llama3' model is pulled.")
    # Depending on the application, might want to raise or have a fallback.
    raise

# Define RAG Prompt Template
PROMPT_TEMPLATE_STR = """
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
rag_prompt = PromptTemplate(template=PROMPT_TEMPLATE_STR, input_variables=["context", "question"])

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


def get_rag_response(user_id: str, question: str, data_source_ids: Optional[List[str]] = None) -> str:
    """
    Retrieves relevant documents and generates a RAG response.
    """
    logger.info(f"Getting RAG response for user '{user_id}', question: '{question[:50]}...'")

    # 1. Retrieve documents
    retrieved_docs = query_vector_db(
        user_id=user_id,
        query=question,
        data_source_ids=data_source_ids,
        n_results=3 # Retrieve 3 documents for context
    )

    if not retrieved_docs:
        logger.warning("No relevant documents found in vector DB for the query.")
        # Optionally, you could still pass to LLM and let it say "I don't know"
        # or return a predefined message.
        # For now, we'll proceed and let the LLM handle the empty context if format_docs_with_sources returns "No context..."
        pass

    # 2. Define the RAG chain using LCEL
    # The chain processes the input dictionary containing 'retrieved_docs' and 'question'.
    rag_chain = (
        RunnableParallel(
            context=(lambda x: format_docs_with_sources(x["retrieved_docs"])),
            question=(lambda x: x["question"])
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Invoke the chain
    try:
        response_text = rag_chain.invoke({"retrieved_docs": retrieved_docs, "question": question})
        logger.info(f"Generated RAG response: {response_text[:100]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error invoking RAG chain: {e}")
        # This might happen if Ollama server is down or model not available at runtime
        return "I'm sorry, but I encountered an error while trying to generate a response."


if __name__ == "__main__":
    logger.info("--- Running RAG Chain Test ---")

    # This test assumes ChromaDB has some data.
    # For a self-contained test, we might add some dummy data first.
    test_user_id = "rag_test_user"
    test_ds_id_rag = "rag_test_source"

    # Add a dummy document for testing this specific RAG chain run
    # This ensures the test doesn't rely on previous state from document_processor.py
    # (though that could also be a valid test scenario)
    logger.info("Adding a dummy document for RAG test...")
    dummy_docs_for_rag = [
        Document(
            page_content="The MochiRAG system is designed for efficient question answering using retrieval augmented generation. It was first conceptualized in early 2024.",
            metadata={"original_filename": "mochi_rag_concept.txt", "page": 1}
        ),
        Document(
            page_content="Key components of MochiRAG include document processing, vector storage, and a query interface. The backend is built with FastAPI.",
            metadata={"original_filename": "mochi_rag_architecture.md", "page": 1}
        )
    ]
    try:
        add_documents_to_vector_db(test_user_id, test_ds_id_rag, dummy_docs_for_rag)
        logger.info("Dummy documents added for RAG test.")
    except Exception as e:
        logger.error(f"Could not add dummy documents for RAG test: {e}. Test might fail or use stale data.")

    # Test case 1: Question relevant to the dummy document
    question1 = "What is the MochiRAG system?"
    logger.info(f"\nTest Case 1: Question: '{question1}' for user '{test_user_id}' using source '{test_ds_id_rag}'")
    try:
        response1 = get_rag_response(test_user_id, question1, data_source_ids=[test_ds_id_rag])
        print(f"\nResponse for Q1:\n{response1}")
    except Exception as e:
        logger.error(f"Error during RAG Test Case 1: {e}")
        print(f"\nError getting response for Q1: {e}")

    # Test case 2: Question not directly in dummy docs (to see "I don't know" or general knowledge)
    question2 = "What is the capital of France?"
    logger.info(f"\nTest Case 2: Question: '{question2}' for user '{test_user_id}' (should ideally not use RAG context heavily)")
    try:
        # Not specifying data_source_ids to see if it picks up the dummy docs or uses general knowledge
        response2 = get_rag_response(test_user_id, question2, data_source_ids=[test_ds_id_rag]) # force using the context
        print(f"\nResponse for Q2:\n{response2}")
    except Exception as e:
        logger.error(f"Error during RAG Test Case 2: {e}")
        print(f"\nError getting response for Q2: {e}")

    # Test case 3: No relevant documents (if we query a different data source id)
    question3 = "Tell me about MochiRAG system."
    non_existent_ds_id = "non_existent_ds_id_123"
    logger.info(f"\nTest Case 3: Question: '{question3}' for user '{test_user_id}' using non-existent source '{non_existent_ds_id}'")
    try:
        response3 = get_rag_response(test_user_id, question3, data_source_ids=[non_existent_ds_id])
        print(f"\nResponse for Q3 (no docs expected):\n{response3}")
    except Exception as e:
        logger.error(f"Error during RAG Test Case 3: {e}")
        print(f"\nError getting response for Q3: {e}")

    # Clean up dummy documents added for this test
    logger.info(f"\nCleaning up dummy documents for user '{test_user_id}' and source '{test_ds_id_rag}'...")
    try:
        from core.vector_store import delete_documents_by_metadata # Re-import for direct call if needed
        delete_documents_by_metadata({"user_id": test_user_id, "data_source_id": test_ds_id_rag})
        logger.info("Dummy documents cleaned up.")
    except Exception as e:
        logger.error(f"Error cleaning up dummy documents: {e}")

    logger.info("--- RAG Chain Test Finished ---")
