from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .llm_manager import llm_manager
from .retriever_manager import retriever_manager
from backend.schemas import QueryRequest, QueryResponse, Source

class RAGChainService:
    def __init__(self):
        # Standard RAG prompt template
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.llm = llm_manager.get_llm()

    def get_rag_response(self, query: QueryRequest, user_id: int) -> QueryResponse:
        retriever = retriever_manager.get_retriever(
            strategy_name=query.strategy,
            user_id=user_id,
            dataset_ids=query.dataset_ids
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Construct the RAG chain using LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Invoke the chain to get the answer
        answer = rag_chain.invoke(query.query)
        
        # Retrieve source documents for citation
        source_docs = retriever.invoke(query.query)
        sources = [Source(page_content=doc.page_content, metadata=doc.metadata) for doc in source_docs]

        return QueryResponse(answer=answer, sources=sources)

# Create a single, globally accessible instance
rag_chain_service = RAGChainService()
