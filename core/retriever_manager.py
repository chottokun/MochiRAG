import logging
from typing import List, Optional, Dict, Any, Literal # Literalをインポート

from langchain_core.retrievers import BaseRetriever
# from langchain_core.embeddings import Embeddings # RetrieverManagerでは直接使わない想定
from langchain_ollama import ChatOllama # 具体的なLLM型として
from langchain_core.language_models import BaseLanguageModel # より汎用的なLLM型として

from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# ParentDocumentRetriever のインポートは戦略クラス内で行うか、必要に応じてここに記述
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever as LangchainParentDocumentRetriever


try:
    from core.config_loader import load_strategy_config, StrategyConfigError
    from core.vector_store_manager import vector_store_manager # VSMのグローバルインスタンス
    from core.embedding_manager import embedding_manager     # EmbeddingManagerのグローバルインスタンス
    from core.llm_manager import llm_manager                 # LLMManagerのグローバルインスタンス
    from core.document_processor import text_splitter as default_text_splitter # ParentDocumentRetrieverで利用想定
except ImportError as e:
    # logger.error(f"Failed to import core managers in retriever_manager: {e}", exc_info=True)
    # フォールバックやエラー処理をここに記述
    # このマネージャは他のマネージャに強く依存するため、フォールバックは限定的
    raise ImportError(f"RetrieverManager failed to import core managers: {e}") from e


logger = logging.getLogger(__name__)

# RAG戦略の型定義 (core.rag_chain.py と共通化するのが望ましい)
RAG_STRATEGY_TYPE = Literal["basic", "parent_document", "multi_query", "contextual_compression"]
AVAILABLE_RAG_STRATEGIES = list(RAG_STRATEGY_TYPE.__args__) # configから読む方が良い

# MultiQueryRetriever用プロンプト (core.rag_chain.py 等と共通化)
# TODO: これもconfigから読み込めるようにする
QUERY_GEN_PROMPT_TEMPLATE_STR = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines.
Original question: {question}"""
try:
    from langchain_core.prompts import PromptTemplate
    query_gen_prompt = PromptTemplate(template=QUERY_GEN_PROMPT_TEMPLATE_STR, input_variables=["question"])
except ImportError:
    query_gen_prompt = None # type: ignore
    logger.warning("Could not import PromptTemplate for MultiQueryRetriever.")

# DeepRag用プロンプトテンプレート
DEEP_RAG_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_STR = """
You are an expert at query decomposition. Your task is to break down a complex user question into simpler, \
atomic sub-questions that can be answered by a retrieval system.
Generate a list of 2 to 4 sub-questions. Each sub-question should be on a new line.

Original Question: {question}

Sub-questions:
"""
try:
    from langchain_core.prompts import PromptTemplate
    deep_rag_query_decomposition_prompt = PromptTemplate(
        template=DEEP_RAG_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_STR,
        input_variables=["question"]
    )
except ImportError:
    deep_rag_query_decomposition_prompt = None # type: ignore
    logger.warning("Could not import PromptTemplate for DeepRag query decomposition.")


class RetrieverStrategyInterface(ABC): # 名前をより明確に
    """リトリーバー戦略のインターフェース"""
    @abstractmethod
    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str, # 使用されたエンベディング戦略の名前
        # vector_store_manager_instance: VectorStoreManager, # VSMはグローバルインスタンスを使う想定に
        # llm_instance: Optional[BaseLanguageModel] = None, # LLMもLLMManagerから取得
        data_source_ids: Optional[List[str]] = None,
        n_results: int = 3,
        **kwargs: Any
    ) -> BaseRetriever:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class BasicRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "basic"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None, n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        embedding_function = embedding_manager.get_embedding_model(embedding_strategy_name)

        # VectorStoreManagerからVectorStoreインスタンスを取得するメソッドを想定
        # ここでは、VSMがChromaクライアントを返す _get_chroma_client を持っていると仮定し、
        # それをラップするか、as_retrieverを持つVectorStoreオブジェクトを返すようにVSMを改修する。
        # 今回は、VSMが直接Chromaクライアントを返すことを仮定して一時的にChromaをインポート。
        try:
            from langchain_chroma import Chroma
        except ImportError:
            logger.error("Langchain_chroma not installed, cannot create Chroma client for BasicRetriever.")
            raise

        # VSMの永続化ディレクトリを使用
        vectorstore = Chroma(
            persist_directory=vector_store_manager.persist_directory,
            embedding_function=embedding_function
        )

        filter_conditions = [{"user_id": user_id}]
        if data_source_ids:
            if len(data_source_ids) == 1:
                filter_conditions.append({"data_source_id": data_source_ids[0]})
            else:
                filter_conditions.append({"data_source_id": {"$in": data_source_ids}})

        final_filter: Optional[Dict[str, Any]] = None
        if filter_conditions:
            final_filter = {"$and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]

        return vectorstore.as_retriever(
            search_kwargs={"k": n_results, "filter": final_filter} if final_filter else {"k": n_results}
        )

class MultiQueryRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "multi_query"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None, n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm() # デフォルトLLMを取得
        if not llm_instance:
            raise ValueError("LLM instance not available via LLMManager for MultiQueryRetriever.")
        if not query_gen_prompt:
            raise ValueError("Query generation prompt not available for MultiQueryRetriever.")

        base_retriever = BasicRetrieverStrategy().get_retriever(
            user_id, embedding_strategy_name, data_source_ids, n_results
        )
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm_instance, prompt=query_gen_prompt
        )

class ContextualCompressionRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "contextual_compression"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None, n_results: int = 5, **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm()
        if not llm_instance:
            raise ValueError("LLM instance not available via LLMManager for ContextualCompressionRetriever.")

        base_retriever = BasicRetrieverStrategy().get_retriever(
            user_id, embedding_strategy_name, data_source_ids, n_results
        )
        compressor = LLMChainExtractor.from_llm(llm_instance)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

class ParentDocumentRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "parent_document"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None, n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        # ParentDocumentRetrieverは child_splitter と vectorstore, docstore が必要
        # child_splitter は document_processor から default_text_splitter を使う想定
        # vectorstore は BasicRetriever と同様に準備
        # docstore は InMemoryStore を使用
        logger.info("Attempting to initialize ParentDocumentRetriever.")
        if not default_text_splitter:
             logger.warning("Default text_splitter not available for ParentDocumentRetriever. Falling back to basic.")
             return BasicRetrieverStrategy().get_retriever(user_id, embedding_strategy_name, data_source_ids, n_results)

        embedding_function = embedding_manager.get_embedding_model(embedding_strategy_name)
        try:
            from langchain_chroma import Chroma
        except ImportError:
            logger.error("Langchain_chroma not installed, cannot create Chroma client for ParentDocumentRetriever.")
            raise

        vectorstore = Chroma(
            persist_directory=vector_store_manager.persist_directory, # VSMの永続化ディレクトリ
            embedding_function=embedding_function,
            # collection_name は user_id や embedding_strategy_name で分けることを検討
        )
        docstore = InMemoryStore() # ParentDocumentRetrieverはdocstoreに親ドキュメントを格納する

        # TODO: ParentDocumentRetriever の正しい動作のためには、
        # ドキュメント追加時に retriever.add_documents(parent_docs) を呼び出す必要がある。
        # 現状の VectorStoreManager はこれに対応していないため、このリトリーバーは期待通りに機能しない可能性が高い。
        # VectorStoreManager が ParentDocumentRetriever の add_documents を呼び出すように拡張するか、
        # または、このリトリーバーを使用する際は特別なインジェスト処理を設ける必要がある。
        logger.warning(
            "ParentDocumentRetriever created, but its effectiveness depends on "
            "how documents were added to the store (requires parent docs in docstore "
            "and child chunks in vectorstore, typically via this retriever's add_documents)."
            "Current setup might not fully support this. Querying child chunks directly."
        )
        # LangchainParentDocumentRetriever の初期化
        # child_splitter には document_processor からインポートした text_splitter を使用
        retriever = LangchainParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=default_text_splitter, # core.document_processor.text_splitter
            # parent_splitter=None, # オプション
            search_kwargs={"k": n_results} # 検索時のパラメータも渡せるように
        )
        # このリトリーバーは、クエリ時にベクトルストアで子チャンクを検索し、
        # docstoreから親ドキュメントを返す。
        # 正しく動作するには、事前にこのリトリーバーの .add_documents() メソッドで
        # 親ドキュメントが処理されている必要がある。
        # 現状、VSMはこれを行わないため、このリトリーバーは主に子チャンクの検索に留まる。
        # より完全な実装のためには、VSM側での対応が必要。
        return retriever

class DeepRagRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "deep_rag"

    def _decompose_query(self, question: str, llm_instance: BaseLanguageModel) -> List[str]:
        """LLMを使って質問をサブクエリに分解する"""
        if not deep_rag_query_decomposition_prompt:
            logger.error("Deep RAG query decomposition prompt is not available.")
            return [question] # フォールバックとして元の質問を返す

        try:
            # TODO: LangChainのLCELを使って書き換える (例: prompt | llm | parser)
            # chain = LLMChain(llm=llm_instance, prompt=deep_rag_query_decomposition_prompt)
            # result = chain.invoke({"question": question})
            # sub_queries_text = result.get("text", "") if isinstance(result, dict) else str(result)

            # LCELを使った書き方 (StrOutputParserを想定)
            from langchain_core.output_parsers import StrOutputParser
            chain = deep_rag_query_decomposition_prompt | llm_instance | StrOutputParser()
            sub_queries_text = chain.invoke({"question": question})

            sub_queries = [q.strip() for q in sub_queries_text.split("\n") if q.strip()]
            logger.info(f"Decomposed question '{question}' into sub-queries: {sub_queries}")
            return sub_queries if sub_queries else [question]
        except Exception as e:
            logger.error(f"Error during query decomposition for DeepRag: {e}", exc_info=True)
            return [question] # エラー時も元の質問にフォールバック

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None, n_results: int = 3, # サブクエリごとの検索結果数
        max_sub_queries: int = 3, # 生成するサブクエリの最大数（プロンプトと合わせる）
        **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm() # クエリ分解用LLM
        if not llm_instance:
            raise ValueError("LLM instance not available for DeepRagRetrieverStrategy query decomposition.")

        # このリトリーバーは _get_relevant_documents をオーバーライドして
        # 内部でクエリ分解と複数検索を行うカスタムリトリーバーとして振る舞う
        # LangChainのBaseRetrieverを継承して実装する

        # カスタムリトリーバーの定義
        class DeepRagCustomRetriever(BaseRetriever):
            user_id: str
            embedding_strategy_name: str
            data_source_ids: Optional[List[str]]
            n_results_per_subquery: int
            max_sub_queries: int
            llm_for_decomposition: BaseLanguageModel
            # base_retriever_strategy: RetrieverStrategyInterface # サブクエリ検索用

            def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
                sub_queries = self._decompose_query_internal(query)

                all_retrieved_docs: List[Document] = []
                doc_ids = set() # 重複排除用

                base_retriever_strategy = BasicRetrieverStrategy() # サブ検索はBasicを使用

                for sub_query in sub_queries[:self.max_sub_queries]:
                    logger.info(f"DeepRAG: Retrieving for sub-query: '{sub_query}'")
                    try:
                        # サブクエリごとにBasicRetrieverを取得して検索
                        # retriever_for_subquery = self.base_retriever_strategy.get_retriever(...)
                        # 上記の代わりに、BasicRetrieverStrategyを直接使う
                        retriever_for_subquery = base_retriever_strategy.get_retriever(
                            user_id=self.user_id,
                            embedding_strategy_name=self.embedding_strategy_name,
                            data_source_ids=self.data_source_ids,
                            n_results=self.n_results_per_subquery
                        )
                        docs = retriever_for_subquery.invoke(sub_query)
                        for doc in docs:
                            # 簡単なIDベースの重複排除 (より高度な重複排除も検討可能)
                            doc_id = doc.metadata.get("data_source_id", "") + "_" + doc.page_content[:50]
                            if doc_id not in doc_ids:
                                all_retrieved_docs.append(doc)
                                doc_ids.add(doc_id)
                    except Exception as e:
                        logger.error(f"DeepRAG: Error retrieving for sub-query '{sub_query}': {e}", exc_info=True)

                logger.info(f"DeepRAG: Retrieved {len(all_retrieved_docs)} unique documents from {len(sub_queries)} sub-queries.")
                # TODO: 必要であれば、ここでさらにドキュメントの再ランキングやフィルタリングを行う
                return all_retrieved_docs

            def _decompose_query_internal(self, question: str) -> List[str]:
                # DeepRagRetrieverStrategyのメソッドを参照する形にする
                # (実際にはこの内部クラスは外部クラスのメソッドを直接呼び出せないため、
                #  コンストラクタで分解用関数自体を渡すなどの工夫が必要になるが、ここでは簡単化)
                #  より良いのは、分解ロジックをこのクラス内に持つか、ヘルパー関数にすること。
                #  ここでは外部クラスのメソッドを呼び出す仮定で進める (実際には修正が必要)
                #  この例では、DeepRagRetrieverStrategyのインスタンスメソッドを直接呼び出すのは構造上問題がある。
                #  代わりに、分解ロジックをこの内部クラスに持たせるか、関数として渡す。
                #  ここでは、外部クラスのメソッドを staticmethod にするか、分解用関数を渡す。
                #  今回は簡単のため、外部クラスのメソッドを呼び出す（これは動作しないので後で修正ポイント）。
                #  → _decompose_query をこのクラスのメソッドとして再定義するのが素直。
                if not deep_rag_query_decomposition_prompt:
                    logger.error("Deep RAG query decomposition prompt is not available.")
                    return [question]
                try:
                    from langchain_core.output_parsers import StrOutputParser
                    chain = deep_rag_query_decomposition_prompt | self.llm_for_decomposition | StrOutputParser()
                    sub_queries_text = chain.invoke({"question": question})
                    sub_queries = [q.strip() for q in sub_queries_text.split("\n") if q.strip()]
                    logger.info(f"DeepRAG (internal): Decomposed question '{question}' into sub-queries: {sub_queries}")
                    return sub_queries if sub_queries else [question]
                except Exception as e:
                    logger.error(f"DeepRAG (internal): Error during query decomposition: {e}", exc_info=True)
                    return [question]


        # カスタムリトリーバーのインスタンス化
        custom_retriever = DeepRagCustomRetriever(
            user_id=user_id,
            embedding_strategy_name=embedding_strategy_name,
            data_source_ids=data_source_ids,
            n_results_per_subquery=n_results,
            max_sub_queries=max_sub_queries,
            llm_for_decomposition=llm_instance,
            # base_retriever_strategy=BasicRetrieverStrategy() # サブ検索用
        )
        return custom_retriever


class RetrieverManager:
    def __init__(self, config_path: Optional[str] = None):
        self.strategies: Dict[str, RetrieverStrategyInterface] = {}
        self.default_strategy_name: Optional[str] = None
        self._load_strategies_from_config(config_path)

    def _load_strategies_from_config(self, config_path: Optional[str] = None):
        try:
            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(f"Failed to load RAG search strategy configuration: {e}.", exc_info=True)
            return

        rag_search_config = config.get("rag_search_strategies", {})
        self.default_strategy_name = rag_search_config.get("default")

        for strat_conf in rag_search_config.get("available", []):
            name = strat_conf.get("name")
            if not name:
                logger.warning(f"Skipping RAG search strategy with no name: {strat_conf}")
                continue

            strategy_instance: Optional[RetrieverStrategyInterface] = None
            if name == "basic":
                strategy_instance = BasicRetrieverStrategy()
            elif name == "multi_query":
                strategy_instance = MultiQueryRetrieverStrategy()
            elif name == "contextual_compression":
                strategy_instance = ContextualCompressionRetrieverStrategy()
            elif name == "parent_document":
                strategy_instance = ParentDocumentRetrieverStrategy()
            elif name == "deep_rag": # DeepRag戦略の登録
                strategy_instance = DeepRagRetrieverStrategy()
            else:
                logger.warning(f"Unsupported RAG search strategy name: {name} from config.")
                continue

            if strategy_instance:
                self.strategies[name] = strategy_instance
                logger.info(f"Registered RAG search strategy: {name}")

        if self.default_strategy_name and self.default_strategy_name not in self.strategies:
            logger.warning(f"Default RAG search strategy '{self.default_strategy_name}' from config not found or failed to load.")
        elif not self.default_strategy_name and self.strategies:
            self.default_strategy_name = list(self.strategies.keys())[0]
            logger.info(f"Using first available RAG search strategy as default: '{self.default_strategy_name}'")


    def get_retriever(
        self,
        name: Optional[RAG_STRATEGY_TYPE] = None,
        user_id: str,
        embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        n_results: Optional[int] = None,
        # DeepRag用のパラメータなどをkwargsで受け取れるようにする
        max_sub_queries: Optional[int] = None,
        n_results_per_subquery: Optional[int] = None,
        **kwargs: Any # その他の戦略特有パラメータ
    ) -> BaseRetriever:
        target_name_str = name if name else self.default_strategy_name
        if not target_name_str:
            raise ValueError("No RAG search strategy name specified and no default is set.")

        # Literal 型への変換を試みる (型安全のため)
        # target_name: RAG_STRATEGY_TYPE
        # if target_name_str not in RAG_STRATEGY_TYPE.__args__: # type: ignore
        #     # 利用可能リストにないが、Literalの定義にはある場合など。通常はconfigのnameと一致するはず。
        #     logger.warning(f"Strategy name '{target_name_str}' is not a predefined RAG_STRATEGY_TYPE. Proceeding with caution.")
        # target_name = cast(RAG_STRATEGY_TYPE, target_name_str)
        # 上記キャストはget_strategyのキーとしては文字列のままが良いので不要かも

        strategy_instance = self.strategies.get(target_name_str)
        if not strategy_instance:
            logger.error(f"RAG search strategy '{target_name_str}' not implemented/registered.")
            if self.default_strategy_name and self.default_strategy_name in self.strategies:
                logger.warning(f"Falling back to default RAG search strategy: {self.default_strategy_name}")
                strategy_instance = self.strategies[self.default_strategy_name]
            else:
                raise ValueError(f"RAG search strategy '{target_name_str}' not found and no default available.")

        # 戦略ごとのパラメータ解決
        # configファイルから戦略固有のデフォルト値を取得し、引数で上書きする
        # 例: n_results
        # config = load_strategy_config() # 再度読むのは非効率なのでManager初期化時に保持する方が良い
        # strategy_configs = {s['name']: s for s in config.get("rag_search_strategies", {}).get("available", [])}
        # current_strat_config = strategy_configs.get(target_name_str, {})

        # final_n_results = n_results if n_results is not None else current_strat_config.get("default_n_results", 3)
        # final_max_sub_queries = max_sub_queries if max_sub_queries is not None else current_strat_config.get("default_max_sub_queries", 3)
        # final_n_results_per_subquery = n_results_per_subquery if n_results_per_subquery is not None else current_strat_config.get("default_n_results_per_subquery", 3)

        # 今回はkwargsで必要なパラメータを渡すことを優先する
        # 各Strategyのget_retrieverが**kwargsを受け取り、必要なものを使う

        # n_results の解決: 引数 > グローバルデフォルト (3)
        # TODO: 設定ファイルから戦略ごとのデフォルトn_resultsを読み込む
        final_n_results = n_results if n_results is not None else 3

        # DeepRag用のパラメータをkwargsに含める
        if target_name_str == "deep_rag":
            if max_sub_queries is not None:
                kwargs["max_sub_queries"] = max_sub_queries
            if n_results_per_subquery is not None:
                kwargs["n_results_per_subquery"] = n_results_per_subquery
            # n_results は DeepRagRetrieverStrategy 内では n_results_per_subquery として解釈される
            # ややこしいので、DeepRagRetrieverStrategyのget_retrieverのシグネチャを n_results_per_subquery に統一する方が良いかも
            # ここでは、n_results をそのまま渡し、DeepRag側で解釈させる。
            # DeepRagRetrieverStrategyのget_retrieverシグネチャは n_results を受け取るので、それがサブクエリごとの取得数として使われる。

        return strategy_instance.get_retriever(
            user_id=user_id,
            embedding_strategy_name=embedding_strategy_name,
            data_source_ids=data_source_ids,
            n_results=final_n_results,
            **kwargs
        )

    def get_available_strategies(self) -> List[str]:
        return list(self.strategies.keys())

retriever_manager = RetrieverManager()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Available RAG search strategies from manager:")
    for name in retriever_manager.get_available_strategies():
        print(f"- {name}")

    # このテストは EmbeddingManager, VectorStoreManager, LLMManager が初期化済みで、
    # テストデータがベクトルストアに存在することを前提とします。
    test_user = "retriever_mgr_test_user"
    # 実際のテストでは、VSMを使って事前にデータを準備する必要があります。
    # 例: global_vector_store_manager.add_documents(...)

    default_emb_strat = embedding_manager.get_available_strategies()[0] if embedding_manager.get_available_strategies() else "dummy_emb_strat"

    if not retriever_manager.get_available_strategies():
        print("No RAG search strategies loaded, cannot run tests.")
    else:
        for strategy_name_str in retriever_manager.get_available_strategies():
            strategy_name: RAG_STRATEGY_TYPE = strategy_name_str # type: ignore

            # LLMを必要とする戦略の場合、LLMが利用可能か確認
            if strategy_name in ["multi_query", "contextual_compression"] and not llm_manager.get_llm():
                print(f"Skipping '{strategy_name}' test as LLM is not available.")
                continue

            print(f"\n--- Testing Retriever Strategy from Manager: {strategy_name} ---")
            try:
                retriever = retriever_manager.get_retriever(
                    name=strategy_name,
                    user_id=test_user,
                    embedding_strategy_name=default_emb_strat, # VSMに追加した際のエンベディング
                )
                print(f"Successfully got retriever for strategy: {strategy_name} -> {type(retriever)}")

                test_query = "What is the MochiRAG system?"
                retrieved_docs = retriever.invoke(test_query)
                print(f"Retrieved {len(retrieved_docs)} documents for query '{test_query[:30]}...' using {strategy_name}:")
                # assert len(retrieved_docs) >= 0 # データがない場合は0件になる
            except Exception as e:
                print(f"Error testing strategy '{strategy_name}': {e}", exc_info=True)

    print("\nRetrieverManager tests finished.")
