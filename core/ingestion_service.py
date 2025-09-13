from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import uuid
import time

from .vector_store_manager import vector_store_manager
from langchain_core.documents import Document
from backend import crud
from backend.database import SessionLocal

class EmbeddingServiceError(Exception):
    """Raised when the embedding service (e.g. Ollama) is unreachable or fails."""


class IngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    def ingest_file(self, file_path: str, file_type: str, data_source_id: int, dataset_id: int, user_id: int, strategy: str = "basic"):
        """
        Ingests a file into the vector store based on the specified strategy.
        """
        if strategy == "parent_document":
            self._ingest_for_parent_document(file_path, file_type, data_source_id, dataset_id, user_id)
        else:
            self._ingest_basic(file_path, file_type, data_source_id, dataset_id, user_id)

    def _ingest_basic(self, file_path: str, file_type: str, data_source_id: int, dataset_id: int, user_id: int):
        """Basic ingestion process."""
        loader = self._get_loader(file_path, file_type)
        if not loader:
            print(f"No suitable loader for file type: {file_type}")
            return

        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        
        for chunk in chunks:
            chunk.metadata.update({
                "data_source_id": data_source_id,
                "dataset_id": dataset_id,
                "user_id": user_id,
                "original_filename": chunk.metadata.get("source", file_path).split('/')[-1]
            })

        collection_name = self._get_collection_name(user_id)

        # Try to add documents with retry/backoff to tolerate temporary embedding service failures
        max_retries = 3
        delay = 1
        for attempt in range(1, max_retries + 1):
            try:
                vector_store_manager.add_documents(collection_name, chunks)
                break
            except Exception as e:
                # If last attempt, raise a domain-specific error for the caller to translate to HTTP 503
                if attempt == max_retries:
                    raise EmbeddingServiceError(f"failed to add documents after {max_retries} attempts: {e}")
                # otherwise wait and retry
                time.sleep(delay)
                delay *= 2

    def _ingest_for_parent_document(self, file_path: str, file_type: str, data_source_id: int, dataset_id: int, user_id: int):
        """Ingestion process for the ParentDocumentRetriever."""
        loader = self._get_loader(file_path, file_type)
        if not loader:
            print(f"No suitable loader for file type: {file_type}")
            return

        docs = loader.load()
        parent_docs = self.parent_splitter.split_documents(docs)
        
        child_docs = []
        db = SessionLocal()
        try:
            for parent_doc in parent_docs:
                parent_id = str(uuid.uuid4())
                parent_doc.metadata.update({
                    "data_source_id": data_source_id,
                    "dataset_id": dataset_id,
                    "user_id": user_id,
                    "original_filename": parent_doc.metadata.get("source", file_path).split('/')[-1],
                })
                
                # Save parent document to the database
                crud.create_parent_document(db, doc_id=parent_id, content=parent_doc.page_content, data_source_id=data_source_id)

                # Create child documents
                sub_docs = self.child_splitter.split_documents([parent_doc])
                for sub_doc in sub_docs:
                    sub_doc.metadata["parent_id"] = parent_id
                child_docs.extend(sub_docs)

            collection_name = self._get_collection_name(user_id)
            # Same retry/backoff strategy for parent/child ingestion
            max_retries = 3
            delay = 1
            for attempt in range(1, max_retries + 1):
                try:
                    vector_store_manager.add_documents(collection_name, child_docs)
                    break
                except Exception as e:
                    if attempt == max_retries:
                        raise EmbeddingServiceError(f"failed to add documents after {max_retries} attempts: {e}")
                    time.sleep(delay)
                    delay *= 2
        finally:
            db.close()


    def _get_loader(self, file_path: str, file_type: str):
        if file_type == 'application/pdf':
            return PyPDFLoader(file_path)
        elif file_type == 'text/plain':
            return TextLoader(file_path)
        elif file_type == 'text/markdown':
            return UnstructuredMarkdownLoader(file_path)
        return None

    def _get_collection_name(self, user_id: int) -> str:
        return f"user_{user_id}"

ingestion_service = IngestionService()