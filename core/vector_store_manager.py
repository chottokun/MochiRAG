import chromadb
from langchain_chroma import Chroma
from typing import List, Optional

from .embedding_manager import embedding_manager
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.embedding_function = embedding_manager.get_embedding_model()

    def initialize_client(self, db_path: str = "chroma_db"):
        """Initializes the ChromaDB client. Should be called once on app startup."""
        if self.client is None:
            print("Initializing ChromaDB client...")
            self.client = chromadb.PersistentClient(path=db_path)
            print("ChromaDB client initialized.")

    def _get_client(self) -> chromadb.Client:
        if self.client is None:
            raise Exception("ChromaDB client is not initialized. Call initialize_client() first.")
        return self.client

    def get_vector_store(self, collection_name: str) -> Chroma:
        """Get a LangChain VectorStore object for a specific collection."""
        return Chroma(
            client=self._get_client(),
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, collection_name: str, documents: List[Document]):
        """Add documents to a specific collection."""
        vector_store = self.get_vector_store(collection_name)
        vector_store.add_documents(documents)
        print(f"{len(documents)} documents added to collection '{collection_name}'.")

    def delete_documents(self, collection_name: str, filter_criteria: dict):
        """Delete documents from a specific collection based on metadata filter."""
        vector_store = self.get_vector_store(collection_name)
        try:
            # Use the 'get' method to find document IDs matching the filter
            # Note: ChromaDB's .get() with a where filter can be inefficient
            # on large datasets if the metadata is not indexed.
            # For this application's scale, it's acceptable.
            ids_to_delete = vector_store.get(where=filter_criteria, include=[])['ids']

            if ids_to_delete:
                vector_store.delete(ids=ids_to_delete)
                print(f"Deleted {len(ids_to_delete)} vectors from collection '{collection_name}'.")
            else:
                print(f"No vectors found to delete in '{collection_name}' with filter: {filter_criteria}")
        except Exception as e:
            # Broad exception for now, but can be refined
            print(f"Error deleting documents from '{collection_name}': {e}")
            # Optionally re-raise or handle specific exceptions
            raise

# Create a single, globally accessible instance
vector_store_manager = VectorStoreManager()
