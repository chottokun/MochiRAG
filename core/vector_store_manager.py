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
    # Fallback for potential circular dependencies or running scripts directly
    # This assumes that if one is missing, the global instances might not be ready.
    # Consider a more robust dependency injection or service locator pattern for larger apps.
    # For now, this allows some level of script execution if paths are tricky.
    if 'global_embedding_manager' not in globals():
        global_embedding_manager = None
    if 'global_chunking_manager' not in globals():
        global_chunking_manager = None


logger = logging.getLogger(__name__)

# ChromaDB永続化パス (vector_store.py と同じ場所を指すようにする)
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
        # Vector DBクライアントは、エンベディング関数を必要とするため、
        # 実際に使用するエンベディング戦略が決まってから初期化されることが多い。
        # ここでは、特定の操作時に都度適切なエンベディング関数で初期化するか、
        # デフォルトのエンベディング関数で初期化しておくかを選択できる。
        # 今回は、操作時に指定されたエンベディング関数でChromaDBクライアントを取得する方式を試みる。
        self._vector_db_clients: Dict[str, Chroma] = {} # embedding_strategy_name -> Chroma client

    def _get_chroma_client(self, embedding_model: Embeddings) -> Chroma:
        """指定されたエンベディングモデルでChromaクライアントを取得または初期化する"""
        # 単純化のため、ここではエンベディングモデルのオブジェクト自体をキーにせず、
        # 永続化ディレクトリは共通とし、ChromaDBが内部でコレクションを分けることを期待する。
        # より厳密には、エンベディングモデルごとにコレクション名を分ける等の工夫が必要。
        # ここでは、embedding_modelのインスタンスに基づいてクライアントをキャッシュすることはせず、
        # 常に新しい（または永続化された）Chromaインスタンスを返す。
        # persist_directory が同じであれば、同じDBを参照する。
        try:
            # logger.info(f"Initializing ChromaDB client with persist_directory: {self.persist_directory} and embedding function: {embedding_model}")
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
        documents: List[Document], # 元の（チャンク分割前）ドキュメント
        embedding_strategy_name: str,
        chunking_strategy_name: str,
        chunking_params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        指定された戦略でドキュメントを処理し、ベクトルストアに追加する。
        Returns: 追加されたチャンクの数
        """
        if not documents:
            logger.info("No documents provided to add.")
            return 0

        # 1. チャンキング戦略を取得し、ドキュメントをチャンク分割
        chunking_strategy = self.chunking_manager.get_strategy(chunking_strategy_name, params=chunking_params)
        doc_chunks = chunking_strategy.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(doc_chunks)} chunks using strategy '{chunking_strategy.get_name()}'.")

        if not doc_chunks:
            logger.info("No chunks were created from the documents.")
            return 0

        # 2. エンベディング戦略とモデルを取得
        embedding_model = self.embedding_manager.get_embedding_model(embedding_strategy_name)

        # 3. メタデータをフィルタリングし、共通メタデータを追加
        # langchain_community.vectorstores.utils.filter_complex_metadata を使用
        filtered_chunks = filter_complex_metadata(doc_chunks)

        for chunk in filtered_chunks:
            if chunk.metadata is None: # filter_complex_metadata はメタデータが存在することを保証するはず
                chunk.metadata = {}
            chunk.metadata["user_id"] = user_id
            chunk.metadata["data_source_id"] = data_source_id
            # 選択された戦略のメタデータも保存しておくと後で役立つ可能性がある
            chunk.metadata["embedding_strategy"] = embedding_strategy_name
            chunk.metadata["chunking_strategy"] = chunking_strategy.get_name() # パラメータ含む名前
            # 元のドキュメント名などもメタデータとして付与されているはず (ローダー依存)

        # 4. ChromaDBクライアントを取得してドキュメントを追加
        try:
            chroma_client = self._get_chroma_client(embedding_model)
            logger.info(f"Adding {len(filtered_chunks)} chunks to ChromaDB for user '{user_id}', source '{data_source_id}' using embedding '{embedding_strategy_name}'.")
            chroma_client.add_documents(documents=filtered_chunks)
            # logger.info("Successfully added documents to ChromaDB. Persisting...")
            # chroma_client.persist() # persist() はChromaのバージョンによって挙動が異なる、または不要な場合がある
            logger.info("Documents added to ChromaDB.")
            return len(filtered_chunks)
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            raise

    def query_documents(
        self,
        user_id: str,
        query: str,
        embedding_strategy_name: str, # 検索時にもエンベディング戦略を指定
        n_results: int = 5,
        data_source_ids: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None # 追加のメタデータフィルタ
    ) -> List[Document]:
        """
        指定されたエンベディング戦略でクエリをベクトル化し、関連ドキュメントを検索する。
        """
        embedding_model = self.embedding_manager.get_embedding_model(embedding_strategy_name)
        chroma_client = self._get_chroma_client(embedding_model)

        # ベースとなるフィルタ条件
        base_filter_conditions = [{"user_id": user_id}]
        if data_source_ids:
            if len(data_source_ids) == 1:
                base_filter_conditions.append({"data_source_id": data_source_ids[0]})
            else:
                base_filter_conditions.append({"data_source_id": {"$in": data_source_ids}})

        # 提供された追加フィルタをマージ
        # ChromaDBの $and 形式に合わせる
        final_filter_list = base_filter_conditions
        if filter_criteria:
            for key, value in filter_criteria.items():
                final_filter_list.append({key: value})

        chroma_filter: Optional[Dict[str, Any]] = None
        if final_filter_list:
            if len(final_filter_list) > 1:
                chroma_filter = {"$and": final_filter_list}
            else: # 条件が1つだけの場合
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

    def delete_documents(self, filter_criteria: Dict[str, Any], embedding_strategy_name: Optional[str] = None) -> bool:
        """
        指定されたメタデータフィルターに一致するドキュメントを削除する。
        特定のエンベディング戦略に関連付けられたクライアントから削除する必要がある場合、
        embedding_strategy_name を指定する。省略された場合はデフォルトエンベディングで試みる。
        """
        if not filter_criteria:
            logger.warning("Deletion filter criteria is empty. No documents will be deleted.")
            return False

        # どのエンベディングモデルのコレクションから削除するか？
        # 現状の実装ではChromaDBクライアントは永続化ディレクトリを共有するため、
        # フィルタだけで特定のドキュメントを狙い撃ちできるはず。
        # ただし、エンベディング関数が異なるとベクトル空間も異なるため、
        # 本来はコレクションを分けて管理するのがより堅牢。
        # ここでは、デフォルトのエンベディングモデルでクライアントを取得して試みる。
        # もしエンベディング戦略ごとにコレクションを分けている場合は、適切なクライアントを選択する必要がある。

        # デフォルトのエンベディング戦略でクライアントを取得
        # (より良いのは、削除対象のドキュメントがどの戦略で追加されたかを知っていること)
        eff_embedding_strategy_name = embedding_strategy_name if embedding_strategy_name else self.embedding_manager.get_available_strategies()[0]
        embedding_model = self.embedding_manager.get_embedding_model(eff_embedding_strategy_name)
        chroma_client = self._get_chroma_client(embedding_model)

        logger.warning(f"Attempting to delete documents with filter: {filter_criteria} using embedding strategy '{eff_embedding_strategy_name}' for client.")

        try:
            # ChromaDBのgetメソッドでIDを取得し、deleteメソッドで削除する
            retrieved_for_delete = chroma_client.get(where=filter_criteria, include=[]) # include=[]でIDのみ取得
            ids_to_delete = retrieved_for_delete.get('ids')

            if ids_to_delete:
                logger.info(f"Found {len(ids_to_delete)} document(s) to delete with IDs: {ids_to_delete}")
                chroma_client.delete(ids=ids_to_delete)
                # chroma_client.persist() # 必要に応じて
                logger.info(f"Successfully deleted {len(ids_to_delete)} document(s) from ChromaDB.")
                return True
            else:
                logger.info("No documents found matching the deletion criteria.")
                return False
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}", exc_info=True)
            raise

# グローバルなVectorStoreManagerインスタンス (シングルトン的に利用も可)
# EmbeddingManagerとChunkingManagerのグローバルインスタンスに依存
vector_store_manager = VectorStoreManager()


if __name__ == '__main__':
    # このテストは、EmbeddingManagerとChunkingManagerが正しく動作することを前提とする
    # また、Ollamaなどの外部サービスが起動している必要がある場合がある

    # 初期化 (グローバルインスタンスが利用可能であると仮定)
    try:
        vsm = VectorStoreManager() # グローバルマネージャを使用
        print("VectorStoreManager initialized successfully.")
    except Exception as e:
        print(f"Error initializing VectorStoreManager: {e}")
        exit()

    test_user = "vsm_test_user"
    test_ds_id = "test_datasource"

    # テスト用ドキュメント
    docs_to_add = [
        Document(page_content="Mochi is a Japanese rice cake made of mochigome.", metadata={"source": "wiki_mochi"}),
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "rag_paper"}),
        Document(page_content="LangChain provides tools for building LLM applications.", metadata={"source": "langchain_docs"})
    ]

    # 利用可能な戦略名を取得
    available_embeddings = global_embedding_manager.get_available_strategies()
    available_chunkers = global_chunking_manager.get_available_strategies()

    if not available_embeddings or not available_chunkers:
        print("No embedding or chunking strategies available. Cannot run test.")
        exit()

    embedding_strat = available_embeddings[0]
    chunking_strat = available_chunkers[0] # デフォルトのRecursive

    print(f"\n--- Testing with Embedding: {embedding_strat}, Chunking: {chunking_strat} ---")

    # 1. ドキュメント追加
    try:
        print(f"Adding documents for user '{test_user}', source '{test_ds_id}'...")
        num_added = vsm.add_documents(
            user_id=test_user,
            data_source_id=test_ds_id,
            documents=docs_to_add,
            embedding_strategy_name=embedding_strat,
            chunking_strategy_name=chunking_strat
        )
        print(f"Successfully added {num_added} chunks to the vector store.")
        assert num_added > 0 # 少なくとも1チャンクは追加されるはず
    except Exception as e:
        print(f"Error adding documents: {e}")
        exit()

    # 2. ドキュメント検索
    try:
        query1 = "What is Mochi?"
        print(f"\nQuerying for: '{query1}'")
        results1 = vsm.query_documents(
            user_id=test_user,
            query=query1,
            embedding_strategy_name=embedding_strat,
            n_results=1,
            data_source_ids=[test_ds_id]
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
            filter_criteria={"source": "rag_paper"} # 追加のメタデータフィルタ
        )
        print(f"Found {len(results2)} result(s):")
        if results2:
            print(f"  - Content: {results2[0].page_content[:50]}..., Metadata: {results2[0].metadata}")
            assert "Retrieval-Augmented Generation" in results2[0].page_content
        else:
            print("No results with that specific source filter, which might be unexpected depending on chunk content.")

    except Exception as e:
        print(f"Error querying documents: {e}")

    # 3. ドキュメント削除
    try:
        print(f"\nDeleting documents for user '{test_user}', source '{test_ds_id}'...")
        delete_filter = {"user_id": test_user, "data_source_id": test_ds_id}
        deleted = vsm.delete_documents(filter_criteria=delete_filter, embedding_strategy_name=embedding_strat)
        print(f"Deletion status: {deleted}")
        assert deleted

        # 削除確認
        results_after_delete = vsm.query_documents(
            user_id=test_user,
            query=query1, # 再度同じクエリ
            embedding_strategy_name=embedding_strat,
            n_results=1,
            data_source_ids=[test_ds_id]
        )
        print(f"Results for '{query1}' after deletion: {len(results_after_delete)}")
        assert len(results_after_delete) == 0
        print("Documents successfully deleted.")

    except Exception as e:
        print(f"Error deleting documents: {e}")

    print("\nVectorStoreManager tests finished.")
