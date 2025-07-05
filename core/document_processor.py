from typing import List, Literal

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import functions from vector_store for the __main__ block
from core.vector_store import add_documents_to_vector_db, query_vector_db, delete_documents_by_metadata


SUPPORTED_FILE_TYPES = Literal["txt", "md", "pdf"]

def load_and_split_document(
    file_path: str,
    file_type: SUPPORTED_FILE_TYPES,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    if file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_type == "md":
        loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    elif file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        supported_types_str = ", ".join(SUPPORTED_FILE_TYPES.__args__)
        raise ValueError(
            f"Unsupported file type: '{file_type}'. Supported types are: {supported_types_str}"
        )

    try:
        loaded_documents = loader.load()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading document {file_path} (type: {file_type}): {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    split_docs = text_splitter.split_documents(loaded_documents)
    return split_docs


if __name__ == "__main__":
    import logging
    from pathlib import Path

    # Setup basic logging for the test script
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sample_docs_dir = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    sample_docs_dir.mkdir(parents=True, exist_ok=True)

    sample_txt_path = sample_docs_dir / "sample.txt"
    sample_md_path = sample_docs_dir / "sample.md"
    sample_pdf_path = sample_docs_dir / "sample.pdf"

    # Ensure sample files exist
    if not sample_txt_path.exists():
        with open(sample_txt_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text document about LangChain and vector stores.\nIt has multiple lines.\n" * 50)
    if not sample_md_path.exists():
        with open(sample_md_path, "w", encoding="utf-8") as f:
            f.write("# Sample Markdown Document\n\nThis is about MochiRAG system.\n\n" * 30)
    if not sample_pdf_path.exists():
        # Assuming PDF was created in a previous step or manually.
        # For this test, if it's missing, we'll just report and skip.
        logger.warning(f"Sample PDF {sample_pdf_path} not found. PDF test will be skipped.")

    test_user_id = "dp_test_user"
    # It's good to use different data_source_ids for different files to test filtering/deletion
    txt_data_source_id = "sample_txt_v1"
    md_data_source_id = "sample_md_v1"
    pdf_data_source_id = "sample_pdf_v1"

    all_processed_docs: List[Document] = []

    logger.info(f"--- Processing {sample_txt_path} ---")
    try:
        txt_docs = load_and_split_document(str(sample_txt_path), "txt")
        logger.info(f"Successfully split sample.txt into {len(txt_docs)} chunks.")
        add_documents_to_vector_db(test_user_id, txt_data_source_id, txt_docs)
        logger.info(f"Added {len(txt_docs)} TXT chunks to vector DB for source '{txt_data_source_id}'.")
        all_processed_docs.extend(txt_docs)
    except Exception as e:
        logger.error(f"Error processing sample.txt: {e}")

    logger.info(f"\n--- Processing {sample_md_path} ---")
    try:
        md_docs = load_and_split_document(str(sample_md_path), "md")
        logger.info(f"Successfully split sample.md into {len(md_docs)} chunks.")
        add_documents_to_vector_db(test_user_id, md_data_source_id, md_docs)
        logger.info(f"Added {len(md_docs)} MD chunks to vector DB for source '{md_data_source_id}'.")
        all_processed_docs.extend(md_docs)
    except Exception as e:
        logger.error(f"Error processing sample.md: {e}")

    if sample_pdf_path.exists():
        logger.info(f"\n--- Processing {sample_pdf_path} ---")
        try:
            pdf_docs = load_and_split_document(str(sample_pdf_path), "pdf")
            logger.info(f"Successfully split sample.pdf into {len(pdf_docs)} chunks.")
            add_documents_to_vector_db(test_user_id, pdf_data_source_id, pdf_docs)
            logger.info(f"Added {len(pdf_docs)} PDF chunks to vector DB for source '{pdf_data_source_id}'.")
            all_processed_docs.extend(pdf_docs)
        except Exception as e:
            logger.error(f"Error processing sample.pdf: {e}")
    else:
        logger.warning(f"Skipping PDF processing as file {sample_pdf_path} does not exist.")

    if all_processed_docs:
        logger.info("\n--- Querying Vector DB ---")
        try:
            query1 = "LangChain vector stores"
            retrieved_txt = query_vector_db(test_user_id, query1, n_results=2, data_source_ids=[txt_data_source_id])
            logger.info(f"Query: '{query1}' (source: {txt_data_source_id}). Retrieved {len(retrieved_txt)} docs:")
            for doc in retrieved_txt:
                logger.info(f"  Content: {doc.page_content[:100]}..., Metadata: {doc.metadata}")

            query2 = "MochiRAG system"
            retrieved_md = query_vector_db(test_user_id, query2, n_results=2, data_source_ids=[md_data_source_id])
            logger.info(f"Query: '{query2}' (source: {md_data_source_id}). Retrieved {len(retrieved_md)} docs:")
            for doc in retrieved_md:
                logger.info(f"  Content: {doc.page_content[:100]}..., Metadata: {doc.metadata}")

            if sample_pdf_path.exists():
                query3 = "Hello PDF" # From the sample PDF content
                retrieved_pdf = query_vector_db(test_user_id, query3, n_results=1, data_source_ids=[pdf_data_source_id])
                logger.info(f"Query: '{query3}' (source: {pdf_data_source_id}). Retrieved {len(retrieved_pdf)} docs:")
                for doc in retrieved_pdf:
                    logger.info(f"  Content: {doc.page_content[:100]}..., Metadata: {doc.metadata}")

            query_all = "document content" # A generic query
            retrieved_all = query_vector_db(test_user_id, query_all, n_results=3) # Query across user's documents
            logger.info(f"Query: '{query_all}' (all user sources). Retrieved {len(retrieved_all)} docs:")
            for doc in retrieved_all:
                logger.info(f"  Content: {doc.page_content[:100]}..., Metadata: {doc.metadata}")


        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
    else:
        logger.warning("No documents were processed and added to the vector DB. Skipping query tests.")

    # Cleanup: Delete the test documents added by this script run
    logger.info("\n--- Cleaning up test data ---")
    try:
        delete_documents_by_metadata({"user_id": test_user_id, "data_source_id": txt_data_source_id})
        logger.info(f"Deleted documents for user '{test_user_id}', source '{txt_data_source_id}'.")
        delete_documents_by_metadata({"user_id": test_user_id, "data_source_id": md_data_source_id})
        logger.info(f"Deleted documents for user '{test_user_id}', source '{md_data_source_id}'.")
        if sample_pdf_path.exists():
            delete_documents_by_metadata({"user_id": test_user_id, "data_source_id": pdf_data_source_id})
            logger.info(f"Deleted documents for user '{test_user_id}', source '{pdf_data_source_id}'.")

        # Verify deletion
        remaining_docs = query_vector_db(test_user_id, "any query", n_results=5)
        if not remaining_docs:
            logger.info("Cleanup successful: No documents found for the test user after deletion.")
        else:
            logger.warning(f"Cleanup check: Found {len(remaining_docs)} documents for test user. Expected 0. Docs: {remaining_docs}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("Document processor test script finished.")
