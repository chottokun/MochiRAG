import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

try:
    from core.embedding_manager import EmbeddingManager, embedding_manager as global_embedding_manager
    from core.chunking_manager import ChunkingManager, chunking_manager as global_chunking_manager
except ImportError:
    if 'global_embedding_manager' not in globals():
        global_embedding_manager = None
    if 'global_chunking_manager' not in globals():
        global_chunking_manager = None


logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

class VectorStoreManager:
    """
    ベクトルストアの操作（ドキュメントの追加、検索、削除）を管理するクラス。
    EmbeddingManager と ChunkingManager を利用してドキュメント処理を行う。
    """
    def __init__(
        self,
        embedding_manager_instance: Optional[EmbeddingManager] = None,
        chunking_manager_instance: Optional[ChunkingManager] = None,
        persist_directory: str = str(CHROMA_PERSIST_DIR)
    ):
        self.embedding_manager = embedding_manager_instance if embedding_manager_instance else global_embedding_manager
        self.chunking_manager = chunking_manager_instance if chunking_manager_instance else global_chunking_manager

        if not self.embedding_manager:
            raise ValueError("EmbeddingManager instance must be provided or globally available.")
        if not self.chunking_manager:
            raise ValueError("ChunkingManager instance must be provided or globally available.")

        self.persist_directory = persist_directory
        self._vector_db_clients: Dict[str, Chroma] = {} 

    def _get_chroma_client(self, embedding_model: Embeddings) -> Chroma:
        try:
            client = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding_model
            )
            return client
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at {self.persist_directory} with embedding model: {e}", exc_info=True)
            raise

    def add_documents(
        self,
        user_id: str,
        data_source_id: str,
        documents: List[Document],
        embedding_strategy_name: str,
        chunking_strategy_name: str,
        chunking_params: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None
    ) -> int:
        if not documents:
            logger.info("No documents provided to add.")
            return 0

        chunking_strategy = self.chunking_manager.get_strategy(chunking_strategy_name, params=chunking_params)
        doc_chunks = chunking_strategy.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(doc_chunks)} chunks using strategy '{chunking_strategy.get_name()}'.")

        if not doc_chunks:
            logger.info("No chunks were created from the documents.")
            return 0

        embedding_model = self.embedding_manager.get_embedding_model(embedding_strategy_name)

        filtered_chunks = filter_complex_metadata(doc_chunks)

        for chunk in filtered_chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["user_id"] = user_id
            chunk.metadata["data_source_id"] = data_source_id
            if dataset_id:
                chunk.metadata["dataset_id"] = dataset_id
            chunk.metadata["embedding_strategy"] = embedding_strategy_name
            chunk.metadata["chunking_strategy"] = chunking_strategy.get_name()

        try:
            chroma_client = self._get_chroma_client(embedding_model)
            logger.info(f"Adding {len(filtered_chunks)} chunks to ChromaDB for user '{user_id}', source '{data_source_id}' using embedding '{embedding_strategy_name}'.")
            chroma_client.add_documents(documents=filtered_chunks)
            logger.info("Documents added to ChromaDB.")
            return len(filtered_chunks)
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            raise

    def query_documents(
        self,
        user_id: str,
        query: str,
        embedding_strategy_name: str,
        n_results: int = 5,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        embedding_model = self.embedding_manager.get_embedding_model(embedding_strategy_name)
        chroma_client = self._get_chroma_client(embedding_model)

        base_filter_conditions = [{"user_id": user_id}]
        if data_source_ids:
            if len(data_source_ids) == 1:
                base_filter_conditions.append({"data_source_id": data_source_ids[0]})
            elif len(data_source_ids) > 1:
                base_filter_conditions.append({"data_source_id": {"$in": data_source_ids}})

        if dataset_ids:
            if len(dataset_ids) == 1:
                base_filter_conditions.append({"dataset_id": dataset_ids[0]})
            elif len(dataset_ids) > 1:
                base_filter_conditions.append({"dataset_id": {"$in": dataset_ids}})

        final_filter_list = base_filter_conditions
        if filter_criteria:
            for key, value in filter_criteria.items():
                final_filter_list.append({key: value})

        chroma_filter: Optional[Dict[str, Any]] = None
        if final_filter_list:
            if len(final_filter_list) > 1:
                chroma_filter = {"$and": final_filter_list}
            elif len(final_filter_list) == 1:
                chroma_filter = final_filter_list[0]

        logger.info(f"Querying ChromaDB for user '{user_id}' with embedding '{embedding_strategy_name}', query: '{query[:50]}...', filter: {chroma_filter}")

        try:
            results = chroma_client.similarity_search(
                query=query,
                k=n_results,
                filter=chroma_filter
            )
            logger.info(f"Retrieved {len(results)} documents from ChromaDB.")
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
            return []

    def delete_documents(self, user_id: str = None, data_source_id: str = None, dataset_id: str = None, filter_criteria: Dict[str, Any] = None, embedding_strategy_name: Optional[str] = None) -> bool:
        """
        指定されたメタデータフィルターに一致するドキュメントを削除する。
        引数の優先順位: filter_criteria > 個別引数 (user_id, data_source_id, dataset_id)
        """
        if filter_criteria:
            # filter_criteriaが指定されている場合はそれを使用（既存の動作を保持）
            final_filter = filter_criteria
        else:
            # 個別引数から filter_criteria を構築
            final_filter = {}
            if user_id:
                final_filter["user_id"] = user_id
            if data_source_id:
                final_filter["data_source_id"] = data_source_id
            if dataset_id:
                final_filter["dataset_id"] = dataset_id

        if not final_filter:
            logger.warning("Deletion filter criteria is empty. No documents will be deleted.")
            return False

        eff_embedding_strategy_name = embedding_strategy_name if embedding_strategy_name else self.embedding_manager.get_available_strategies()[0]
        embedding_model = self.embedding_manager.get_embedding_model(eff_embedding_strategy_name)
        chroma_client = self._get_chroma_client(embedding_model)

        logger.warning(f"Attempting to delete documents with filter: {final_filter} using embedding strategy '{eff_embedding_strategy_name}' for client.")

        try:
            retrieved_for_delete = chroma_client.get(where=final_filter, include=[])
            ids_to_delete = retrieved_for_delete.get("ids")

            if ids_to_delete:
                logger.info(f"Found {len(ids_to_delete)} document(s) to delete with IDs: {ids_to_delete}")
                chroma_client.delete(ids=ids_to_delete)
                logger.info(f"Successfully deleted {len(ids_to_delete)} document(s) from ChromaDB.")
                return True
            else:
                logger.info("No documents found matching the deletion criteria.")
                return False
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}", exc_info=True)
            raise

vector_store_manager = VectorStoreManager()


if __name__ == '__main__':
    try:
        vsm = VectorStoreManager()
        print("VectorStoreManager initialized successfully.")
    except Exception as e:
        print(f"Error initializing VectorStoreManager: {e}")
        exit()

    test_user = "vsm_test_user"
    test_ds_id = "test_datasource"
    test_dataset_id = "test_dataset_uuid"

    docs_to_add = [
        Document(page_content="Mochi is a Japanese rice cake made of mochigome.", metadata={"source": "wiki_mochi"}),
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "rag_paper"}),
        Document(page_content="LangChain provides tools for building LLM applications.", metadata={"source": "langchain_docs"})
    ]

    available_embeddings = global_embedding_manager.get_available_strategies()
    available_chunkers = global_chunking_manager.get_available_strategies()

    if not available_embeddings or not available_chunkers:
        print("No embedding or chunking strategies available. Cannot run test.")
        exit()

    embedding_strat = available_embeddings[0]
    chunking_strat = available_chunkers[0]

    print(f"\n--- Testing with Embedding: {embedding_strat}, Chunking: {chunking_strat} ---")

    try:
        print(f"Adding documents for user '{test_user}', source '{test_ds_id}'...")
        num_added = vsm.add_documents(
            user_id=test_user,
            data_source_id=test_ds_id,
            documents=docs_to_add,
            embedding_strategy_name=embedding_strat,
            chunking_strategy_name=chunking_strat,
            dataset_id=test_dataset_id
        )
        print(f"Successfully added {num_added} chunks to the vector store.")
        assert num_added > 0
    except Exception as e:
        print(f"Error adding documents: {e}")
        exit()

    try:
        query1 = "What is Mochi?"
        print(f"\nQuerying for: '{query1}'")
        results1 = vsm.query_documents(
            user_id=test_user,
            query=query1,
            embedding_strategy_name=embedding_strat,
            n_results=1,
            data_source_ids=[test_ds_id],
            dataset_ids=[test_dataset_id]
        )
        print(f"Found {len(results1)} result(s):")
        for res_doc in results1:
            print(f"  - Content: {res_doc.page_content[:50]}..., Metadata: {res_doc.metadata}")
        assert len(results1) > 0
        assert "mochi" in results1[0].page_content.lower()

        query2 = "Tell me about RAG."
        print(f"\nQuerying for: '{query2}' with specific source filter")
        results2 = vsm.query_documents(
            user_id=test_user,
            query=query2,
            embedding_strategy_name=embedding_strat,
            n_results=1,
            filter_criteria={"source": "rag_paper"},
            dataset_ids=[test_dataset_id]
        )
        print(f"Found {len(results2)} result(s):")
        if results2:
            print(f"  - Content: {results2[0].page_content[:50]}..., Metadata: {results2[0].metadata}")
            assert "Retrieval-Augmented Generation" in results2[0].page_content
        else:
            print("No results with that specific source filter, which might be unexpected depending on chunk content.")

    except Exception as e:
        print(f"Error querying documents: {e}")

    try:
        print(f"\nDeleting documents for user '{test_user}', source '{test_ds_id}'...")
        deleted = vsm.delete_documents(user_id=test_user, data_source_id=test_ds_id, dataset_id=test_dataset_id, embedding_strategy_name=embedding_strat)
        print(f"Deletion status: {deleted}")
        assert deleted

        results_after_delete = vsm.query_documents(
            user_id=test_user,
            query=query1,
            embedding_strategy_name=embedding_strat,
            n_results=1,
            data_source_ids=[test_ds_id],
            dataset_ids=[test_dataset_id]
        )
        print(f"Results for '{query1}' after deletion: {len(results_after_delete)}")
        assert len(results_after_delete) == 0
        print("Documents successfully deleted.")

    except Exception as e:
        print(f"Error deleting documents: {e}")

    print("\nVectorStoreManager tests finished.")

