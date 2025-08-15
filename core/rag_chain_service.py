import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .llm_manager import llm_manager
from .retriever_manager import retriever_manager
from .deep_rag_strategy import DeepRAGStrategy
from backend.schemas import QueryRequest, QueryResponse, Source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_RAG_PROMPT_TEMPLATE_STR = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Try to keep the answer concise and informative.
When you use information from the context, cite the source using the metadata (e.g., "According to [document_name from metadata], ...").
The 'document_name' can be found in the 'original_filename' or 'data_source_id' fields of the source metadata.

Context:
{context}

Question: {question}

Answer:
"""
default_rag_prompt = PromptTemplate(template=DEFAULT_RAG_PROMPT_TEMPLATE_STR, input_variables=["context", "question"])

def format_docs_with_sources(docs: List[Document]) -> str:
    if not docs:
        return "No context documents found."

    formatted_docs = []
    for i, doc in enumerate(docs):
        source_info = "Unknown Source"
        if doc.metadata:
            original_filename = doc.metadata.get("original_filename")
            data_source_id = doc.metadata.get("data_source_id")
            if original_filename and data_source_id:
                source_info = f"Source (ID: {data_source_id}, Original: {original_filename})"
            elif data_source_id:
                source_info = f"Source (ID: {data_source_id})"
            elif original_filename:
                 source_info = f"Source (Original: {original_filename})"
            page_num = doc.metadata.get("page")
            if page_num is not None:
                source_info += f", Page: {page_num + 1}"

        formatted_docs.append(f"{source_info}\nContent: {doc.page_content}")

    return "\n\n---\n\n".join(formatted_docs)

class RAGChainService:
    def get_rag_response(self, query: QueryRequest, user_id: int) -> QueryResponse:
        logger.info(f"Getting RAG response for user '{user_id}', rag_strategy: '{query.strategy}', question: '{query.query[:50]}...'")

        if query.strategy == 'deeprag':
            deep_rag = DeepRAGStrategy(user_id=user_id, dataset_ids=query.dataset_ids)
            result = deep_rag.run(question=query.query)
            # The trace from DeepRAG is a list of dictionaries.
            # We can format this into the Source schema.
            sources = []
            for step in result.get("trace", []):
                # Each step in the trace becomes a source
                source_content = f"Sub-query: {step['subquery']}\nIntermediate Answer: {step['answer']}"
                # Include retrieved docs for the step as metadata
                step_sources = [f"{doc.metadata.get('original_filename', 'N/A')}: {doc.page_content[:100]}..." for doc in step['sources']]
                source_metadata = {"step": step, "step_sources": step_sources}
                sources.append(Source(page_content=source_content, metadata=source_metadata))
            return QueryResponse(answer=result['answer'], sources=sources)

        try:
            retriever = retriever_manager.get_retriever(
                strategy_name=query.strategy,
                user_id=user_id,
                dataset_ids=query.dataset_ids
            )
        except Exception as e:
            logger.error(f"Failed to get retriever for strategy '{query.strategy}': {e}", exc_info=True)
            return QueryResponse(
                answer=f"I'm sorry, but I encountered an error setting up the retrieval mechanism for strategy '{query.strategy}'.",
                sources=[]
            )

        retrieve_and_format_chain = RunnableParallel(
            retrieved_docs=RunnableLambda(lambda x: x["question"]) | retriever,
            question_passthrough=RunnablePassthrough()
        ).assign(
            formatted_context=lambda x: format_docs_with_sources(x["retrieved_docs"])
        )

        rag_chain_with_sources = (
            retrieve_and_format_chain
            | RunnableParallel(
                answer=(
                    RunnableLambda(lambda x: {"context": x["formatted_context"], "question": x["question_passthrough"]["question"]})
                    | default_rag_prompt
                    | llm_manager.get_llm()
                    | StrOutputParser()
                ),
                sources=RunnableLambda(lambda x: x["retrieved_docs"])
            )
        )

        try:
            result = rag_chain_with_sources.invoke({"question": query.query})
            logger.info(f"Generated RAG response: {result['answer'][:100]}..., Sources count: {len(result['sources'])}")
            
            sources = [Source(page_content=doc.page_content, metadata=doc.metadata) for doc in result['sources']]
            return QueryResponse(answer=result['answer'], sources=sources)

        except Exception as e:
            logger.error(f"Error invoking RAG chain with strategy '{query.strategy}': {e}", exc_info=True)
            return QueryResponse(
                answer=f"I'm sorry, but I encountered an error while trying to generate a response using the '{query.strategy}' strategy.",
                sources=[]
            )

rag_chain_service = RAGChainService()
