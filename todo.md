backend/main.py

The delete_document_from_dataset endpoint only deletes the document's metadata from the SQL database. It does not remove the corresponding vectors from the vector store (ChromaDB). This leaves orphaned data in the vector store, which can be retrieved in future queries, leading to incorrect answers and potential data leakage.

To fix this, you should:

Add a delete_documents method to core/vector_store_manager.py that can remove vectors based on a metadata filter.
Call this new method from this endpoint before deleting the record from the SQL database.

-----

core/vector_store_manager.py

The VectorStoreManager is missing a method to delete documents from ChromaDB. This is a critical feature for data lifecycle management and is required by the delete_document API endpoint. Without it, deleting a document only removes its metadata, leaving orphaned vectors in the database, which can lead to data leakage and incorrect query results. Please add a delete_documents method that can remove vectors based on metadata filters.

sample

```python
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
            ids_to_delete = vector_store.get(where=filter_criteria, include=["metadatas"])['ids']
            if ids_to_delete:
                vector_store.delete(ids=ids_to_delete)
                print(f"Deleted {len(ids_to_delete)} documents from collection '{collection_name}'.")
            else:
                print(f"No documents found to delete in collection '{collection_name}' with filter: {filter_criteria}")
        except Exception as e:
            print(f"An error occurred while deleting documents from '{collection_name}': {e}")
            raise

# Create a single, globally accessible instance
vector_store_manager = VectorStoreManager()
```