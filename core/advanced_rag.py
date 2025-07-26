import logging
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseRAGStrategy

logger = logging.getLogger(__name__)

class AdvancedRAGStrategy(BaseRAGStrategy):
    """
    高度なRAG戦略の実装。
    複数段階の検索、リランキング、複数のプロンプトテンプレートなどを使用する。
    """
    
    def get_name(self) -> str:
        return "advanced_rag"

    def retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]:
        """
        高度な検索戦略を実装。
        初期検索 -> リランキング -> 最終選択
        """
        try:
            from core.vector_store_manager import vector_store_manager
        except ImportError:
            logger.error("vector_store_manager could not be imported")
            return []
        
        # 初期検索では多めに取得
        initial_k = self.config.get("initial_retrieval_k", 10)
        final_k = self.config.get("final_retrieval_k", 5)
        
        logger.info(f"Initial retrieval of {initial_k} documents for question: {question[:50]}...")
        
        # 初期検索
        initial_docs = vector_store_manager.query_documents(
            user_id=user_id,
            query=question,
            embedding_strategy_name=embedding_strategy_for_retrieval,
            n_results=initial_k,
            data_source_ids=data_source_ids
        )
        
        if not initial_docs:
            logger.warning("No documents retrieved in initial search")
            return []
        
        # リランキング（簡単な実装例）
        reranked_docs = self._rerank_documents(question, initial_docs)
        
        # 最終的な文書数に絞り込み
        final_docs = reranked_docs[:final_k]
        
        logger.info(f"Final selection: {len(final_docs)} documents after reranking")
        return final_docs
    
    def _rerank_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """
        文書のリランキングを行う。
        ここでは簡単な例として、質問との類似度に基づいてソートする。
        実際の実装では、専用のリランキングモデルを使用することが多い。
        """
        use_reranker = self.config.get("use_reranker", False)
        
        if not use_reranker:
            # リランキングを使用しない場合は、そのまま返す
            return documents
        
        # 簡単なキーワードベースのスコアリング例
        question_words = set(question.lower().split())
        
        def calculate_score(doc: Document) -> float:
            doc_words = set(doc.page_content.lower().split())
            common_words = question_words.intersection(doc_words)
            return len(common_words) / len(question_words) if question_words else 0
        
        # スコアに基づいてソート
        scored_docs = [(doc, calculate_score(doc)) for doc in documents]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Documents reranked based on keyword similarity")
        return [doc for doc, score in scored_docs]

    def generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]:
        """
        高度な応答生成。
        複数のプロンプトテンプレートを試行し、最適な応答を選択する。
        """
        llm_model_name = self.config.get("llm_model_name", "gpt-4")
        temperature = self.config.get("temperature", 0.1)
        
        logger.info(f"Generating response using advanced strategy with model: {llm_model_name}")
        
        llm = ChatOpenAI(model=llm_model_name, temperature=temperature)
        
        # 複数のプロンプトテンプレートを定義
        templates = self.config.get("prompt_templates", [
            {
                "name": "detailed",
                "template": (
                    "以下のコンテキストを詳細に分析し、質問に対して包括的な回答を提供してください。\n"
                    "コンテキスト: {context}\n"
                    "質問: {question}\n"
                    "詳細な回答:"
                )
            },
            {
                "name": "concise", 
                "template": (
                    "以下のコンテキストから要点を抽出し、質問に対して簡潔に答えてください。\n"
                    "コンテキスト: {context}\n"
                    "質問: {question}\n"
                    "簡潔な回答:"
                )
            }
        ])
        
        # デフォルトテンプレートを使用する場合
        if not templates:
            templates = [{
                "name": "default",
                "template": (
                    "以下のコンテキストのみを使用して、質問に答えてください。\n"
                    "コンテキスト: {context}\n"
                    "質問: {question}\n"
                    "回答:"
                )
            }]
        
        # 使用するテンプレートを選択（設定で指定、デフォルトは最初のもの）
        selected_template_name = self.config.get("selected_template", templates[0]["name"])
        selected_template = next(
            (t for t in templates if t["name"] == selected_template_name), 
            templates[0]
        )
        
        prompt = ChatPromptTemplate.from_template(selected_template["template"])
        
        # コンテキストを構築
        context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        
        # チェーンを実行
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Generated response using template '{selected_template_name}', length: {len(answer)}")
        
        return {
            "answer": answer, 
            "sources": retrieved_documents,
            "template_used": selected_template_name
        }

