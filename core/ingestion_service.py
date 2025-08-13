from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

from .vector_store_manager import vector_store_manager
from langchain_core.documents import Document

class IngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def ingest_file(self, file_path: str, file_type: str, data_source_id: int, dataset_id: int, user_id: int):
        """Ingests a file into the vector store."""
        loader = self._get_loader(file_path, file_type)
        if not loader:
            print(f"No suitable loader for file type: {file_type}")
            return

        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update({
                "data_source_id": data_source_id,
                "dataset_id": dataset_id,
                "user_id": user_id,
                "original_filename": chunk.metadata.get("source", file_path).split('/')[-1]
            })

        # The collection name will be the user's ID to ensure data separation
        collection_name = self._get_collection_name(user_id)
        vector_store_manager.add_documents(collection_name, chunks)

    def _get_loader(self, file_path: str, file_type: str):
        if file_type == 'application/pdf':
            return PyPDFLoader(file_path)
        elif file_type == 'text/plain':
            return TextLoader(file_path)
        elif file_type == 'text/markdown':
            return UnstructuredMarkdownLoader(file_path)
        return None

    def _get_collection_name(self, user_id: int) -> str:
        # We use the user_id to create a unique and isolated collection for each user.
        return f"user_{user_id}"

# Create a single, globally accessible instance
ingestion_service = IngestionService()
