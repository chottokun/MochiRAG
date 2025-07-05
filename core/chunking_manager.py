from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker # SemanticChunkerをインポート
from langchain_core.embeddings import Embeddings # SemanticChunkerがEmbeddingsを要求するため

import logging

try:
    from core.config_loader import load_strategy_config, StrategyConfigError
    from core.embedding_manager import embedding_manager # SemanticChunkerがEmbeddingモデルインスタンスを必要とするため
except ImportError:
    # logger.warning("Could not import config_loader or embedding_manager in chunking_manager. Using fallback defaults.")
    def load_strategy_config(): # フォールバック用のダミー関数
        return {
            "chunking_strategies": {
                "default": "recursive_cs1000_co200",
                "available": [
                    {"name": "recursive_cs1000_co200", "type": "recursive_text_splitter", "params": {"chunk_size": 1000, "chunk_overlap": 200}}
                ]
            }
        }
    class StrategyConfigError(Exception): pass
    embedding_manager = None # フォールバック

logger = logging.getLogger(__name__)

# デフォルト値はconfigファイルからロードされるべきだが、ここにも定義しておく
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class ChunkingStrategy(ABC):
    """チャンキング戦略のインターフェース"""
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """ドキュメントを指定された戦略でチャンクに分割する"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """戦略名を返す"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """戦略の設定パラメータを返す"""
        pass

class RecursiveTextSplitterChunking(ChunkingStrategy):
    """RecursiveCharacterTextSplitterを使用したチャンキング戦略"""
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        return self._splitter.split_documents(documents)

    def get_name(self) -> str:
        return f"recursive_text_splitter_cs{self.chunk_size}_co{self.chunk_overlap}"

    def get_config(self) -> Dict[str, Any]:
        return {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap, "type": "recursive_text_splitter"}

class SemanticChunkingStrategy(ChunkingStrategy):
    """SemanticChunkerを使用したチャンキング戦略"""
    def __init__(self, embedding_model_instance: Embeddings, breakpoint_threshold_type: str = "percentile", **kwargs: Any):
        self.embedding_model_instance = embedding_model_instance
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.kwargs = kwargs # その他のSemanticChunkerのパラメータ
        try:
            self._splitter = SemanticChunker(
                embeddings=self.embedding_model_instance,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                **self.kwargs
            )
            logger.info(f"SemanticChunkingStrategy initialized with threshold type: {self.breakpoint_threshold_type}")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticChunker: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize SemanticChunker: {e}") from e

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        # SemanticChunkerは langchain_core.documents.Document のリストを期待する
        return self._splitter.split_documents(documents)

    def get_name(self) -> str:
        # 動的な名前生成も可能だが、configファイルでの 'name' を使うことを想定
        return f"semantic_chunker_btt_{self.breakpoint_threshold_type}"

    def get_config(self) -> Dict[str, Any]:
        # embedding_model_instance は実行時オブジェクトなのでconfigには含めにくい
        # 代わりに、どのembedding_strategyを参照したかを保存する方が良いかもしれない
        return {"breakpoint_threshold_type": self.breakpoint_threshold_type, "type": "semantic_chunker", **self.kwargs}


class ChunkingManager:
    """チャンキング戦略を管理するクラス。設定ファイルから動的に戦略をロードする。"""
    def __init__(self, config_path: Optional[str] = None):
        self.strategies: Dict[str, ChunkingStrategy] = {}
        self.default_strategy_name: Optional[str] = None
        self._load_strategies_from_config(config_path)

    def _load_strategies_from_config(self, config_path: Optional[str] = None):
        try:
            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(f"ChunkingManager: Failed to load strategy configuration: {e}. No strategies will be available.", exc_info=True)
            return

        chunking_config = config.get("chunking_strategies")
        if not isinstance(chunking_config, dict):
            logger.warning("ChunkingManager: 'chunking_strategies' section not found or invalid in config. No strategies loaded.")
            return

        self.default_strategy_name = chunking_config.get("default")
        available_configs = chunking_config.get("available")

        if not isinstance(available_configs, list):
            logger.warning("ChunkingManager: 'chunking_strategies.available' section not found or not a list. No strategies loaded.")
            return

        for strat_config in available_configs:
            if not isinstance(strat_config, dict):
                logger.warning(f"ChunkingManager: Skipping invalid strategy config item (not a dict): {strat_config}")
                continue

            name = strat_config.get("name")
            strat_type = strat_config.get("type") # 'type' を 'strat_type' に

            if not all([name, strat_type]): # paramsはオプションなのでチェックしない
                logger.warning(f"ChunkingManager: Skipping incomplete chunking strategy config: {strat_config} (missing name or type).")
                continue

            params = strat_config.get("params", {}) # paramsがなくても空のdictを渡す
            strategy_instance: Optional[ChunkingStrategy] = None
            try:
                if strat_type == "recursive_text_splitter":
                    strategy_instance = RecursiveTextSplitterChunking(
                        chunk_size=params.get("chunk_size", DEFAULT_CHUNK_SIZE),
                        chunk_overlap=params.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                    )
                elif strat_type == "semantic_chunker":
                    embedding_strategy_ref = strat_config.get("embedding_strategy_ref")
                    if not embedding_strategy_ref:
                        logger.warning(f"ChunkingManager: SemanticChunker strategy '{name}' requires 'embedding_strategy_ref'. Skipping.")
                        continue
                    if not embedding_manager:
                        logger.error("ChunkingManager: EmbeddingManager is not available for SemanticChunker. Skipping.")
                        continue
                    try:
                        embedding_model_instance = embedding_manager.get_embedding_model(embedding_strategy_ref)
                        # SemanticChunkerが受け付けるパラメータを明示的に指定
                        semantic_init_params = {}
                        if "breakpoint_threshold_type" in params:
                            semantic_init_params["breakpoint_threshold_type"] = params["breakpoint_threshold_type"]
                        # 他にSemanticChunkerが受け付けるパラメータがあればここに追加
                        # 例: if "add_start_index" in params:
                        #         semantic_init_params["add_start_index"] = params["add_start_index"]

                        strategy_instance = SemanticChunkingStrategy(
                            embedding_model_instance=embedding_model_instance,
                            **semantic_init_params # フィルタリングされたパラメータのみを渡す
                        )
                    except Exception as emb_e:
                        logger.error(f"ChunkingManager: Failed to get embedding model or init SemanticChunker for '{name}': {emb_e}", exc_info=True)
                        continue
                else:
                    logger.warning(f"ChunkingManager: Unsupported chunking strategy type: {strat_type} for strategy '{name}'")
                    continue

                if strategy_instance:
                    self.strategies[name] = strategy_instance
                    logger.info(f"ChunkingManager: Successfully registered chunking strategy: {name}")

            except Exception as e:
                logger.error(f"ChunkingManager: Failed to initialize or register chunking strategy '{name}': {e}", exc_info=True)

        if self.default_strategy_name and self.default_strategy_name not in self.strategies:
            logger.warning(f"ChunkingManager: Default chunking strategy '{self.default_strategy_name}' not found. Default will be unset.")
            self.default_strategy_name = None

        if not self.default_strategy_name and self.strategies:
            self.default_strategy_name = list(self.strategies.keys())[0]
            logger.info(f"ChunkingManager: No valid default chunking strategy specified. Using first available: '{self.default_strategy_name}'")
        elif not self.strategies:
            logger.warning("ChunkingManager: No chunking strategies were loaded or registered.")


    def get_strategy(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> ChunkingStrategy:
        """
        指定された名前またはパラメータに基づいてチャンキング戦略を取得する。
        paramsが優先され、該当するtypeの新しいインスタンスを生成しようと試みる。
        nameのみの場合、登録済みの戦略を返す。どちらもない場合はデフォルト戦略を返す。
        """
        if params: # APIからchunk_sizeなどが直接指定された場合
            # この場合、nameは基本タイプ（例：'recursive_text_splitter'）を期待
            base_type_name = name or "recursive_text_splitter" # nameがなければrecursiveを仮定
            if base_type_name == "recursive_text_splitter":
                cs = params.get("chunk_size", DEFAULT_CHUNK_SIZE)
                co = params.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                # 動的に生成される戦略名はキャッシュしない（または別の管理方法を検討）
                return RecursiveTextSplitterChunking(chunk_size=cs, chunk_overlap=co)
            elif base_type_name == "semantic_chunker":
                embedding_strategy_ref = params.get("embedding_strategy_ref") # APIから指定される想定
                if not embedding_strategy_ref or not embedding_manager:
                    raise ValueError("Semantic chunking requires 'embedding_strategy_ref' and available EmbeddingManager.")
                try:
                    embedding_model_instance = embedding_manager.get_embedding_model(embedding_strategy_ref)
                    return SemanticChunkingStrategy(
                        embedding_model_instance=embedding_model_instance,
                        breakpoint_threshold_type=params.get("breakpoint_threshold_type", "percentile"),
                        **(params.get("additional_params", {}))
                    )
                except Exception as e:
                    raise ValueError(f"Could not create SemanticChunkingStrategy with params {params}: {e}") from e

            logger.warning(f"Dynamic creation for chunking type '{base_type_name}' with params not fully supported or type unknown. Falling back.")


        target_name = name if name else self.default_strategy_name
        if not target_name:
            raise ValueError("No chunking strategy name provided and no default strategy is set.")

        strategy = self.strategies.get(target_name)
        if not strategy:
            # 設定ファイルに定義された名前で見つからなかった場合、パラメータを見て動的生成を試みる
            # （例：recursive_cs500_co50 のような名前だが、設定ファイルにはない場合）
            if target_name.startswith("recursive_text_splitter_cs"):
                try:
                    parts = target_name.split('_')
                    cs = int(parts[-2][2:]) # csXXX
                    co = int(parts[-1][2:]) # coYYY
                    logger.info(f"Dynamically creating RecursiveTextSplitterChunking for '{target_name}'")
                    return RecursiveTextSplitterChunking(chunk_size=cs, chunk_overlap=co)
                except Exception:
                    logger.warning(f"Could not dynamically parse params from strategy name '{target_name}'.")

            logger.error(f"Chunking strategy '{target_name}' not found. Available: {list(self.strategies.keys())}")
            if self.default_strategy_name and self.default_strategy_name in self.strategies:
                logger.warning(f"Falling back to default strategy: {self.default_strategy_name}")
                return self.strategies[self.default_strategy_name]
            raise ValueError(f"Chunking strategy '{target_name}' not found.")
        return strategy

    def get_available_strategies(self) -> List[str]:
        return list(self.strategies.keys())

# グローバルなChunkingManagerインスタンス
chunking_manager = ChunkingManager()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Available chunking strategies from manager (loaded from config):")
    for name in chunking_manager.get_available_strategies():
        print(f"- {name}")

    # テストドキュメントの準備
    doc1 = Document(page_content="This is the first document. It is fairly short.")
    doc2 = Document(page_content="This is the second document, and it is a bit longer to see splitting. " * 50)
    sample_docs = [doc1, doc2]

    # デフォルト戦略のテスト
    default_strategy_name = RecursiveTextSplitterChunking().get_name()
    print(f"\nTesting default strategy: {default_strategy_name}")
    strategy1 = chunking_manager.get_strategy(default_strategy_name)
    chunks1 = strategy1.split_documents(sample_docs)
    print(f"Number of chunks from strategy '{strategy1.get_name()}': {len(chunks1)}")
    if chunks1:
        print(f"First chunk example: '{chunks1[0].page_content[:100]}...'")
    print(f"Strategy config: {strategy1.get_config()}")

    # パラメータ指定での戦略取得テスト
    custom_params = {"chunk_size": 100, "chunk_overlap": 20}
    custom_strategy_name_base = "recursive_text_splitter" # 実際の名前はパラメータに依存する
    print(f"\nTesting strategy '{custom_strategy_name_base}' with params: {custom_params}")
    # このget_strategyはパラメータから新しいインスタンスを生成する
    strategy2 = chunking_manager.get_strategy(custom_strategy_name_base + f"_cs{custom_params['chunk_size']}_co{custom_params['chunk_overlap']}", params=custom_params)
    chunks2 = strategy2.split_documents(sample_docs)
    print(f"Number of chunks from strategy '{strategy2.get_name()}': {len(chunks2)}")
    if chunks2:
        print(f"First chunk example: '{chunks2[0].page_content[:100]}...'")
    print(f"Strategy config: {strategy2.get_config()}")


    # # SemanticChunkerのテスト例 (HuggingFaceEmbeddingsとlangchain_experimentalが必要)
    # # このテストは、必要なライブラリがインストールされている場合にのみ実行されるようにする
    # try:
    #     from langchain_huggingface import HuggingFaceEmbeddings
    #     from langchain_experimental.text_splitter import SemanticChunker
    #     test_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #     semantic_chunker_name = SemanticChunkingStrategy(embedding_model=test_embeddings).get_name()
    #     # セマンティックチャンカーをマネージャーに手動で追加（通常は_register_default_strategiesで行う）
    #     if semantic_chunker_name not in chunking_manager.get_available_strategies():
    #          chunking_manager.add_strategy(SemanticChunkingStrategy(embedding_model=test_embeddings))
    #
    #     print(f"\nTesting strategy: {semantic_chunker_name}")
    #     strategy3 = chunking_manager.get_strategy(semantic_chunker_name)
    #     # SemanticChunkerは通常、より大きなドキュメントで効果を発揮
    #     large_doc_content = " ".join(["This is sentence one."] * 20 + ["This is sentence two, quite different."] * 20)
    #     semantic_docs = [Document(page_content=large_doc_content)]
    #     chunks3 = strategy3.split_documents(semantic_docs)
    #     print(f"Number of chunks from strategy '{strategy3.get_name()}': {len(chunks3)}")
    #     if chunks3:
    #         for i, chunk in enumerate(chunks3):
    #             print(f"  Chunk {i+1}: '{chunk.page_content[:100]}...'")
    #     print(f"Strategy config: {strategy3.get_config()}")
    # except ImportError:
    #     print("\nSkipping SemanticChunker test: langchain_experimental or langchain_huggingface not fully available.")
    # except Exception as e:
    #     print(f"\nError during SemanticChunker test: {e}")

    print("\nChunkingManager tests finished.")
