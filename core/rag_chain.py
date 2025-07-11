import logging
from typing import List, Optional, Dict, Any, Literal

# Langchain component imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever as LangchainParentDocumentRetriever


# Project-specific imports
try:
    from core.retriever_manager import retriever_manager, RAG_STRATEGY_TYPE, AVAILABLE_RAG_STRATEGIES
    from core.embedding_manager import embedding_manager
    from core.llm_manager import llm_manager # LLMManagerのインスタンスを使用
except ImportError as e:
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from core.retriever_manager import retriever_manager, RAG_STRATEGY_TYPE, AVAILABLE_RAG_STRATEGIES
        from core.embedding_manager import embedding_manager
        from core.llm_manager import llm_manager
    except ImportError as ie:
        # logger.error(f"Failed to import core managers in rag_chain.py: {ie}")
        raise


# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLMインスタンスはLLMManagerから必要に応じて取得する
# llm = llm_manager.get_llm() # モジュールロード時のグローバルllm取得は削除

# RAG_STRATEGY_TYPE と AVAILABLE_RAG_STRATEGIES は retriever_manager からインポート済

# --- Prompt Templates ---
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

# For MultiQueryRetriever
QUERY_GEN_PROMPT_TEMPLATE_STR = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines.
Original question: {question}"""
query_gen_prompt = PromptTemplate(template=QUERY_GEN_PROMPT_TEMPLATE_STR, input_variables=["question"])


# --- Document Formatting ---
def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Formats a list of documents into a single string for RAG context,
    including source information from metadata.
    """
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
            # Add page number if available
            page_num = doc.metadata.get("page")
            if page_num is not None:
                source_info += f", Page: {page_num + 1}" # PyPDFLoader pages are 0-indexed

        formatted_docs.append(f"{source_info}\nContent: {doc.page_content}")

    return "\n\n---\n\n".join(formatted_docs)


# --- Main RAG Function ---

def get_rag_response(
    user_id: str,
    question: str,
    data_source_ids: Optional[List[str]] = None,
    dataset_ids: Optional[List[str]] = None, # 追加: データセットIDリスト
    rag_strategy: RAG_STRATEGY_TYPE = "basic",
    embedding_strategy_for_retrieval: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves relevant documents using the specified RAG strategy and generates a response.
    Returns a dictionary containing the answer and the list of retrieved source documents.
    Utilizes the RetrieverManager to get the appropriate retriever.
    """
    logger.info(f"Getting RAG response for user '{user_id}', rag_strategy: '{rag_strategy}', embedding_for_retrieval: '{embedding_strategy_for_retrieval}', question: '{question[:50]}...'")

    if not embedding_strategy_for_retrieval:
        # フォールバックまたはエラー処理: API側で設定されるべき
        logger.warning("embedding_strategy_for_retrieval not provided to get_rag_response. Using default from EmbeddingManager.")
        embedding_strategy_for_retrieval = embedding_manager.get_available_strategies()[0]


    # 1. RetrieverManager を使用してリトリーバーを取得
    try:
        retriever = retriever_manager.get_retriever(
            name=rag_strategy,
            user_id=user_id,
            embedding_strategy_name=embedding_strategy_for_retrieval,
            data_source_ids=data_source_ids,
            dataset_ids=dataset_ids # 追加
            # n_results は各リトリーバー戦略のデフォルト、または RetrieverManager で設定可能
        )
    except Exception as e:
        logger.error(f"Failed to get retriever for strategy '{rag_strategy}': {e}", exc_info=True)
        return {
            "answer": f"I'm sorry, but I encountered an error setting up the retrieval mechanism for strategy '{rag_strategy}'.",
            "sources": []
        }

    # 2. Define the RAG chain using LCEL
    #    The retriever is now obtained from RetrieverManager

    # Define a sub-chain to retrieve and format documents
    # This will output a dictionary with 'retrieved_docs' and 'formatted_context'
    retrieve_and_format_chain = RunnableParallel(
        retrieved_docs=RunnableLambda(lambda x: x["question"]) | retriever,
        question_passthrough=RunnablePassthrough() # Pass the original question through for the next step
    ).assign(
        formatted_context=lambda x: format_docs_with_sources(x["retrieved_docs"])
    )

    # Main RAG chain
    rag_chain_with_sources = (
        retrieve_and_format_chain
        | RunnableParallel(
            answer=(
                RunnableLambda(lambda x: {"context": x["formatted_context"], "question": x["question_passthrough"]["question"]})
                | default_rag_prompt
                | llm_manager.get_llm()
                | StrOutputParser()
            ),
            sources=RunnableLambda(lambda x: x["retrieved_docs"]) # Pass retrieved_docs to the output
        )
    )

    # 3. Invoke the chain
    try:
        # The input to the chain should be a dictionary containing the question
        result = rag_chain_with_sources.invoke({"question": question})
        logger.info(f"Generated RAG response: {result['answer'][:100]}..., Sources count: {len(result['sources'])}")
        return result # This will be a dict like {"answer": "...", "sources": [...docs...]}
    except Exception as e:
        logger.error(f"Error invoking RAG chain with strategy '{rag_strategy}': {e}", exc_info=True)
        # Return a dict in case of error as well, to match the expected return type
        return {
            "answer": f"I'm sorry, but I encountered an error while trying to generate a response using the '{rag_strategy}' strategy.",
            "sources": []
        }

# if __name__ == "__main__":
#     # このブロックのテストは、各マネージャのユニットテストやE2Eテストでカバーされるため、
#     # 維持コストを考慮して削除またはコメントアウトを推奨。
#     # logger.info("--- Running RAG Chain Test with Strategies (using global managers) ---")
#     # ... (テストコード) ...
#     pass
