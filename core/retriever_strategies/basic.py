from typing import List, Optional, Any, Dict
from langchain_core.retrievers import BaseRetriever
from core.embedding_manager import embedding_manager
from core.vector_store_manager import vector_store_manager
from langchain_chroma import Chroma
from core.retriever_strategies.interface import RetrieverStrategyInterface
import logging

logger = logging.getLogger(__name__)

class BasicRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "basic"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        embedding_function = embedding_manager.get_embedding_model(embedding_strategy_name)

        try:
            vectorstore = Chroma(
                persist_directory=vector_store_manager.persist_directory,
                embedding_function=embedding_function
            )
        except ImportError:
            logger.error("Langchain_chroma not installed, cannot create Chroma client for BasicRetriever.")
            raise

        filter_conditions = [{"user_id": user_id}]
        if data_source_ids:
            if len(data_source_ids) == 1:
                filter_conditions.append({"data_source_id": data_source_ids[0]})
            elif len(data_source_ids) > 1:
                filter_conditions.append({"data_source_id": {"$in": data_source_ids}})

        if dataset_ids:
            if len(dataset_ids) == 1:
                filter_conditions.append({"dataset_id": dataset_ids[0]})
            elif len(dataset_ids) > 1:
                filter_conditions.append({"dataset_id": {"$in": dataset_ids}})

        final_filter: Optional[Dict[str, Any]] = None
        if filter_conditions:
            if len(filter_conditions) > 1:
                final_filter = {"$and": filter_conditions}
            elif len(filter_conditions) == 1:
                final_filter = filter_conditions[0]

        search_kwargs = {"k": n_results}
        if final_filter:
            search_kwargs["filter"] = final_filter

        return vectorstore.as_retriever(search_kwargs=search_kwargs)