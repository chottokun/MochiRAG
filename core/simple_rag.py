import logging
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseRAGStrategy

logger = logging.getLogger(__name__)

class SimpleRAGStrategy(BaseRAGStrategy):
    """
    シンプルなRAG戦略の実装。
    ベクトル検索でドキュメントを取得し、プロンプトテンプレートを使用してLLMで応答を生成する。
    """
    
    def get_name(self) -> str:
        return "simple_rag"

    def retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]:
        """
        ベクトルストアから関連ドキュメントを検索する。
        """
        try:
            from core.vector_store_manager import vector_store_manager
        except ImportError:
            logger.error("vector_store_manager could not be imported")
            return []
        
        # 設定から検索数を取得、デフォルトは4
        n_results = self.config.get("retrieval_k", 4)
        
        logger.info(f"Retrieving {n_results} documents for question: {question[:50]}...")
        
        retrieved_docs = vector_store_manager.query_documents(
            user_id=user_id,
            query=question,
            embedding_strategy_name=embedding_strategy_for_retrieval,
            n_results=n_results,
            data_source_ids=data_source_ids
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]:
        """
        検索されたドキュメントを使用してLLMで応答を生成する。
        """
        # LLMとプロンプトテンプレートの設定
        llm_model_name = self.config.get("llm_model_name", "gpt-3.5-turbo")
        temperature = self.config.get("temperature", 0)
        
        logger.info(f"Generating response using model: {llm_model_name}")
        
        llm = ChatOpenAI(model=llm_model_name, temperature=temperature)

        # プロンプトテンプレートを設定から取得、デフォルトを使用
        template = self.config.get("prompt_template", 
            "以下のコンテキストのみを使用して、質問に答えてください。\n"
            "コンテキスト: {context}\n"
            "質問: {question}\n"
            "回答:"
        )
        
        prompt = ChatPromptTemplate.from_template(template)

        # コンテキストを構築
        context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        
        # チェーンを実行
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Generated response of length: {len(answer)}")
        
        return {"answer": answer, "sources": retrieved_documents}

