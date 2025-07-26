import logging
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseRAGStrategy

logger = logging.getLogger(__name__)

class DeepRAGStrategy(BaseRAGStrategy):
    """
    自律型RAG戦略の実装（DeepRAG）。
    複数のRAGコンポーネントを動的に組み合わせ、推論時に最適なパスを選択する。
    """
    
    def get_name(self) -> str:
        return "deep_rag"

    def execute(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]:
        """
        DeepRAGの自律的な実行フロー。
        質問の複雑さを分析し、適切な戦略を動的に選択する。
        """
        logger.info(f"Starting DeepRAG execution for question: {question[:50]}...")
        
        # 1. 質問の複雑さを分析
        complexity_analysis = self._analyze_question_complexity(question)
        
        # 2. 複雑さに基づいて実行戦略を決定
        execution_strategy = self._determine_execution_strategy(complexity_analysis)
        
        logger.info(f"Question complexity: {complexity_analysis['level']}, Strategy: {execution_strategy}")
        
        # 3. 選択された戦略に基づいて実行
        if execution_strategy == "simple":
            return self._execute_simple_flow(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        elif execution_strategy == "multi_hop":
            return self._execute_multi_hop_flow(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        elif execution_strategy == "iterative":
            return self._execute_iterative_flow(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        else:
            # フォールバック
            return self._execute_simple_flow(user_id, question, data_source_ids, embedding_strategy_for_retrieval)

    def _analyze_question_complexity(self, question: str) -> Dict[str, Any]:
        """
        質問の複雑さを分析する。
        """
        # 簡単な複雑さ分析の例
        question_lower = question.lower()
        
        # 複雑さの指標
        multi_part_indicators = ["and", "or", "but", "however", "また", "さらに", "一方"]
        comparison_indicators = ["compare", "contrast", "difference", "similar", "比較", "違い", "類似"]
        temporal_indicators = ["when", "before", "after", "during", "いつ", "前", "後", "間"]
        causal_indicators = ["why", "because", "cause", "reason", "なぜ", "理由", "原因"]
        
        complexity_score = 0
        indicators_found = []
        
        # 複数部分の質問
        if any(indicator in question_lower for indicator in multi_part_indicators):
            complexity_score += 2
            indicators_found.append("multi_part")
        
        # 比較を求める質問
        if any(indicator in question_lower for indicator in comparison_indicators):
            complexity_score += 2
            indicators_found.append("comparison")
        
        # 時間的な要素
        if any(indicator in question_lower for indicator in temporal_indicators):
            complexity_score += 1
            indicators_found.append("temporal")
        
        # 因果関係
        if any(indicator in question_lower for indicator in causal_indicators):
            complexity_score += 1
            indicators_found.append("causal")
        
        # 質問の長さも考慮
        if len(question.split()) > 15:
            complexity_score += 1
            indicators_found.append("long")
        
        # 複雑さレベルを決定
        if complexity_score >= 4:
            level = "high"
        elif complexity_score >= 2:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": complexity_score,
            "indicators": indicators_found
        }

    def _determine_execution_strategy(self, complexity_analysis: Dict[str, Any]) -> str:
        """
        複雑さ分析に基づいて実行戦略を決定する。
        """
        level = complexity_analysis["level"]
        indicators = complexity_analysis["indicators"]
        
        # 設定から戦略選択のルールを取得
        strategy_rules = self.config.get("strategy_rules", {
            "high_complexity": "iterative",
            "medium_complexity": "multi_hop", 
            "low_complexity": "simple"
        })
        
        if level == "high":
            return strategy_rules.get("high_complexity", "iterative")
        elif level == "medium":
            # 比較や複数部分の質問の場合はmulti_hopを優先
            if "comparison" in indicators or "multi_part" in indicators:
                return "multi_hop"
            return strategy_rules.get("medium_complexity", "multi_hop")
        else:
            return strategy_rules.get("low_complexity", "simple")

    def _execute_simple_flow(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]:
        """
        シンプルなRAGフローを実行。
        """
        logger.info("Executing simple RAG flow")
        
        retrieved_documents = self.retrieve(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        response = self.generate_response(question, retrieved_documents)
        response["execution_strategy"] = "simple"
        return response

    def _execute_multi_hop_flow(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]:
        """
        マルチホップRAGフローを実行。
        複数回の検索と推論を行う。
        """
        logger.info("Executing multi-hop RAG flow")
        
        max_hops = self.config.get("max_hops", 3)
        all_retrieved_docs = []
        reasoning_steps = []
        
        current_question = question
        
        for hop in range(max_hops):
            logger.info(f"Multi-hop iteration {hop + 1}/{max_hops}")
            
            # 現在の質問で検索
            hop_docs = self.retrieve(user_id, current_question, data_source_ids, embedding_strategy_for_retrieval)
            all_retrieved_docs.extend(hop_docs)
            
            if not hop_docs:
                logger.info(f"No documents found in hop {hop + 1}, stopping")
                break
            
            # 中間推論を実行
            if hop < max_hops - 1:  # 最後のホップでない場合
                intermediate_response = self._generate_intermediate_reasoning(current_question, hop_docs)
                reasoning_steps.append({
                    "hop": hop + 1,
                    "question": current_question,
                    "reasoning": intermediate_response,
                    "docs_count": len(hop_docs)
                })
                
                # 次の質問を生成
                current_question = self._generate_follow_up_question(question, intermediate_response)
                if not current_question:
                    break
        
        # 最終応答を生成
        final_response = self.generate_response(question, all_retrieved_docs)
        final_response["execution_strategy"] = "multi_hop"
        final_response["reasoning_steps"] = reasoning_steps
        final_response["total_hops"] = len(reasoning_steps) + 1
        
        return final_response

    def _execute_iterative_flow(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]:
        """
        反復的RAGフローを実行。
        応答の品質を評価し、必要に応じて改善を行う。
        """
        logger.info("Executing iterative RAG flow")
        
        max_iterations = self.config.get("max_iterations", 3)
        quality_threshold = self.config.get("quality_threshold", 0.7)
        
        best_response = None
        best_quality = 0
        iteration_history = []
        
        for iteration in range(max_iterations):
            logger.info(f"Iterative RAG iteration {iteration + 1}/{max_iterations}")
            
            # 検索パラメータを調整（反復ごとに異なる戦略を試す）
            adjusted_k = self.config.get("retrieval_k", 5) + iteration
            
            # 検索実行
            retrieved_docs = self._retrieve_with_params(
                user_id, question, data_source_ids, embedding_strategy_for_retrieval, 
                n_results=adjusted_k
            )
            
            # 応答生成
            response = self.generate_response(question, retrieved_docs)
            
            # 応答品質を評価
            quality_score = self._evaluate_response_quality(question, response["answer"], retrieved_docs)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "quality_score": quality_score,
                "docs_count": len(retrieved_docs),
                "response_length": len(response["answer"])
            })
            
            # 最良の応答を更新
            if quality_score > best_quality:
                best_quality = quality_score
                best_response = response
                best_response["quality_score"] = quality_score
            
            # 品質閾値に達した場合は早期終了
            if quality_score >= quality_threshold:
                logger.info(f"Quality threshold reached at iteration {iteration + 1}")
                break
        
        if best_response is None:
            # フォールバック
            best_response = self._execute_simple_flow(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        
        best_response["execution_strategy"] = "iterative"
        best_response["iteration_history"] = iteration_history
        best_response["final_quality_score"] = best_quality
        
        return best_response

    def retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]:
        """
        標準的な検索実装。
        """
        return self._retrieve_with_params(user_id, question, data_source_ids, embedding_strategy_for_retrieval)

    def _retrieve_with_params(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str, n_results: Optional[int] = None) -> List[Document]:
        """
        パラメータ指定可能な検索実装。
        """
        try:
            from core.vector_store_manager import vector_store_manager
        except ImportError:
            logger.error("vector_store_manager could not be imported")
            return []
        
        if n_results is None:
            n_results = self.config.get("retrieval_k", 5)
        
        retrieved_docs = vector_store_manager.query_documents(
            user_id=user_id,
            query=question,
            embedding_strategy_name=embedding_strategy_for_retrieval,
            n_results=n_results,
            data_source_ids=data_source_ids
        )
        
        return retrieved_docs

    def generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]:
        """
        標準的な応答生成。
        """
        llm_model_name = self.config.get("llm_model_name", "gpt-4")
        temperature = self.config.get("temperature", 0.2)
        
        llm = ChatOpenAI(model=llm_model_name, temperature=temperature)
        
        template = self.config.get("prompt_template",
            "以下のコンテキストを使用して、質問に対して正確で詳細な回答を提供してください。\n"
            "コンテキスト: {context}\n"
            "質問: {question}\n"
            "回答:"
        )
        
        prompt = ChatPromptTemplate.from_template(template)
        context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {"answer": answer, "sources": retrieved_documents}

    def _generate_intermediate_reasoning(self, question: str, documents: List[Document]) -> str:
        """
        中間推論を生成する。
        """
        llm_model_name = self.config.get("llm_model_name", "gpt-4")
        llm = ChatOpenAI(model=llm_model_name, temperature=0.1)
        
        template = (
            "以下の文書を分析し、質問に関連する重要な情報を抽出してください。\n"
            "文書: {context}\n"
            "質問: {question}\n"
            "重要な情報:"
        )
        
        prompt = ChatPromptTemplate.from_template(template)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
        return response.content if hasattr(response, 'content') else str(response)

    def _generate_follow_up_question(self, original_question: str, intermediate_reasoning: str) -> Optional[str]:
        """
        フォローアップ質問を生成する。
        """
        llm_model_name = self.config.get("llm_model_name", "gpt-4")
        llm = ChatOpenAI(model=llm_model_name, temperature=0.3)
        
        template = (
            "元の質問と中間推論に基づいて、さらに詳しい情報を得るためのフォローアップ質問を生成してください。\n"
            "元の質問: {original_question}\n"
            "中間推論: {intermediate_reasoning}\n"
            "フォローアップ質問:"
        )
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = prompt | llm
        response = chain.invoke({
            "original_question": original_question,
            "intermediate_reasoning": intermediate_reasoning
        })
        
        follow_up = response.content if hasattr(response, 'content') else str(response)
        
        # 空の応答や無意味な応答の場合はNoneを返す
        if len(follow_up.strip()) < 10:
            return None
        
        return follow_up.strip()

    def _evaluate_response_quality(self, question: str, answer: str, sources: List[Document]) -> float:
        """
        応答の品質を評価する（0.0-1.0のスコア）。
        """
        # 簡単な品質評価の例
        quality_score = 0.0
        
        # 1. 応答の長さ（適度な長さが良い）
        answer_length = len(answer.split())
        if 20 <= answer_length <= 200:
            quality_score += 0.3
        elif 10 <= answer_length <= 300:
            quality_score += 0.2
        
        # 2. ソースの活用度（ソースの内容が応答に含まれているか）
        if sources:
            source_content = " ".join([doc.page_content for doc in sources])
            source_words = set(source_content.lower().split())
            answer_words = set(answer.lower().split())
            
            if source_words and answer_words:
                overlap_ratio = len(source_words.intersection(answer_words)) / len(source_words)
                quality_score += min(overlap_ratio * 0.4, 0.4)
        
        # 3. 質問への関連性
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if question_words and answer_words:
            relevance_ratio = len(question_words.intersection(answer_words)) / len(question_words)
            quality_score += min(relevance_ratio * 0.3, 0.3)
        
        return min(quality_score, 1.0)

