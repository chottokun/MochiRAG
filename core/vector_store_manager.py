import chromadb
from langchain_chroma import Chroma
from typing import List, Optional
import logging

from .embedding_manager import embedding_manager
from .config_manager import config_manager
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.embedding_function = embedding_manager.get_embedding_model()

    def initialize_client(self):
        """
        Initializes the ChromaDB client based on the configuration
        in `config/strategies.yaml`. Should be called once on app startup.
        """
        if self.client is not None:
            return

        logger.info("Initializing ChromaDB client...")
        config = config_manager.get_vector_store_config()

        if config.mode == 'http':
            logger.info(f"Connecting to ChromaDB server at {config.host}:{config.port}...")
            self.client = chromadb.HttpClient(host=config.host, port=config.port)
        elif config.mode == 'persistent':
            logger.info(f"Initializing persistent ChromaDB at path: {config.path}")
            self.client = chromadb.PersistentClient(path=config.path)
        else:
            raise ValueError(f"Unsupported ChromaDB mode: {config.mode}")

        logger.info("ChromaDB client initialized successfully.")

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
        logger.info(f"{len(documents)} documents added to collection '{collection_name}'.")

# Create a single, globally accessible instance
vector_store_manager = VectorStoreManager()
