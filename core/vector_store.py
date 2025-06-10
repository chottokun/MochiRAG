from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path # Ensure Path is imported at the top

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata # Import the filter utility

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding function
try:
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        # You can specify cache_folder to control where models are downloaded
        # cache_folder="./sentence_transformer_cache/"
    )
    logger.info("SentenceTransformerEmbeddings loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformerEmbeddings: {e}. Ensure a working internet connection for first download or check model name.")
    raise

# Define ChromaDB persistence path
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

try:
    vector_db_client = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_function
    )
    logger.info(f"ChromaDB initialized. Data will be persisted to: {CHROMA_PERSIST_DIR}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB at {CHROMA_PERSIST_DIR}: {e}")
    raise


def add_documents_to_vector_db(
    user_id: str, data_source_id: str, documents: List[Document]
) -> None:
    if not documents:
        logger.info("No documents provided to add to vector DB.")
        return

    # Filter complex metadata before adding user_id and data_source_id
    filtered_documents = filter_complex_metadata(documents)

    for doc in filtered_documents:
        if doc.metadata is None: # Should not happen after filter_complex_metadata if it ensures metadata exists
            doc.metadata = {}
        doc.metadata["user_id"] = user_id
        doc.metadata["data_source_id"] = data_source_id

    try:
        logger.info(f"Adding {len(filtered_documents)} documents to ChromaDB for user '{user_id}', source '{data_source_id}'.")
        vector_db_client.add_documents(documents=filtered_documents)
        vector_db_client.persist() # Note: Deprecated warning may appear
        logger.info("Successfully added documents and persisted ChromaDB.")
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        raise

def query_vector_db(
    user_id: str, query: str, n_results: int = 5, data_source_ids: Optional[List[str]] = None
) -> List[Document]:
    """
    Queries the ChromaDB vector store for documents relevant to the user's query,
    filtered by user_id and optionally by a list of data_source_ids.
    """
    # Start with base conditions for the filter
    conditions = [{"user_id": user_id}]

    if data_source_ids:
        if len(data_source_ids) == 1:
            conditions.append({"data_source_id": data_source_ids[0]})
        else:
            conditions.append({"data_source_id": {"$in": data_source_ids}})

    # Construct the final filter for similarity_search
    final_filter: Dict[str, Any]
    if len(conditions) > 1:
        final_filter = {"$and": conditions}
    elif conditions: # Should always be true because user_id is always present
        final_filter = conditions[0] # A single condition like {"user_id": "some_id"}
    else:
        # This case should ideally not be reached if user_id is mandatory for querying.
        # If user_id could be optional, this might return unfiltered results or all user data.
        # For safety, if no conditions, perhaps default to a non-matching filter or log error.
        logger.warning("Querying without any filters. This might return broad results.")
        final_filter = {}


    try:
        logger.info(f"Querying ChromaDB for user '{user_id}' with query: '{query[:50]}...' and filter: {final_filter}")
        results = vector_db_client.similarity_search(
            query=query,
            k=n_results,
            filter=final_filter
        )
        logger.info(f"Retrieved {len(results)} documents from ChromaDB.")
        return results
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return []

def delete_documents_by_metadata(filter_criteria: Dict[str, Any]) -> None:
    logger.warning(f"Attempting to delete documents with filter: {filter_criteria}")
    try:
        if not filter_criteria:
            logger.warning("Deletion filter criteria is empty. No documents will be deleted.")
            return

        and_conditions = []
        for key, value in filter_criteria.items():
            and_conditions.append({key: value})

        final_filter: Dict[str, Any]
        if len(and_conditions) > 1:
            final_filter = {"$and": and_conditions}
        elif len(and_conditions) == 1:
            final_filter = and_conditions[0]
        else:
            logger.warning("No valid conditions derived from filter_criteria for deletion.")
            return

        logger.info(f"Constructed deletion filter for ChromaDB 'get' for IDs: {final_filter}")

        retrieved_for_delete = vector_db_client.get(where=final_filter, include=[])
        ids_to_delete = retrieved_for_delete.get('ids')

        if ids_to_delete:
            logger.info(f"Found {len(ids_to_delete)} document(s) to delete with IDs: {ids_to_delete}")
            vector_db_client.delete(ids=ids_to_delete)
            vector_db_client.persist()
            logger.info(f"Successfully deleted {len(ids_to_delete)} document(s) and persisted ChromaDB.")
        else:
            logger.info("No documents found matching the deletion criteria.")

    except Exception as e:
        logger.error(f"Error deleting documents from ChromaDB: {e}")
        raise

if __name__ == "__main__":
    logger.info("Running basic tests for vector_store.py...")

    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"page": 1}),
        Document(page_content="A B C D E F G.", metadata={"page": 2}),
        Document(page_content="Exploring vector databases and their applications.", metadata={"page": 1}),
    ]

    test_user_id = "test_user_123"
    test_data_source_id = "sample_source_v1"

    try:
        add_documents_to_vector_db(test_user_id, test_data_source_id, docs)
        logger.info("Test documents added successfully.")
    except Exception as e:
        logger.error(f"Failed to add test documents: {e}")
        exit()

    try:
        query_text = "fox"
        # Test query with both user_id and data_source_id
        retrieved_docs = query_vector_db(test_user_id, query_text, n_results=2, data_source_ids=[test_data_source_id])
        logger.info(f"Query 1: '{query_text}' with data_source_id. Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"  Doc {i+1}: Content='{doc.page_content}', Metadata={doc.metadata}")

        if any("fox" in doc.page_content for doc in retrieved_docs):
            logger.info("Query 1 test successful: Found relevant document.")
        else:
            logger.warning("Query 1 test: Did not find expected document containing 'fox'.")

        # Test query with only user_id
        retrieved_docs_user_only = query_vector_db(test_user_id, "vector databases", n_results=1)
        logger.info(f"Query 2: 'vector databases' (user_id only). Retrieved {len(retrieved_docs_user_only)} documents:")
        for i, doc in enumerate(retrieved_docs_user_only):
            logger.info(f"  Doc {i+1}: Content='{doc.page_content}', Metadata={doc.metadata}")
        if any("vector databases" in doc.page_content for doc in retrieved_docs_user_only):
            logger.info("Query 2 test successful: Found relevant document.")
        else:
            logger.warning("Query 2 test: Did not find expected document for 'vector databases'.")


    except Exception as e:
        logger.error(f"Failed to query documents: {e}")

    try:
        logger.info(f"Attempting to delete documents for user_id '{test_user_id}' and data_source_id: '{test_data_source_id}'")
        delete_documents_by_metadata({"user_id": test_user_id, "data_source_id": test_data_source_id})

        retrieved_after_delete = query_vector_db(test_user_id, query_text, n_results=2, data_source_ids=[test_data_source_id])

        if not retrieved_after_delete:
            logger.info("Deletion test successful: No documents found for the user and source after deletion.")
        else:
            logger.warning(f"Deletion test: Found {len(retrieved_after_delete)} documents after attempting deletion. Expected 0 for this user/source. Docs: {retrieved_after_delete}")

    except Exception as e:
        logger.error(f"Failed during deletion test: {e}")

    logger.info("Basic tests for vector_store.py finished.")
