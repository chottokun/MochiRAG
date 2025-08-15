from typing import List, Tuple, Any, Dict
from langchain_core.prompts import PromptTemplate

from .llm_manager import llm_manager
from .vector_store_manager import vector_store_manager

class DeepRAGStrategy:
    def __init__(self, user_id: int, dataset_ids: List[int]):
        self.llm = llm_manager.get_llm()
        collection_name = f"user_{user_id}"
        self.vectorstore = vector_store_manager.get_vector_store(collection_name)
        self.dataset_ids = dataset_ids

        self.decomp_prompt = PromptTemplate(
            input_variables=["question", "history"],
            template=(
                "You are a RAG system that decomposes queries step-by-step.\n"
                "History: {history}\n"
                "Main Question: {question}\n"
                "Generate the next atomic subquery or 'TERMINATE' to finish."
            )
        )
        self.answer_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Use the following context to answer the query.\n"
                "Context: {context}\n"
                "Query: {query}\n"
                "Answer concisely."
            )
        )

    def run(self, question: str, max_depth: int = 5) -> Dict[str, Any]:
        final_answer, trace = self._binary_tree_search(question, max_depth)
        return {"answer": final_answer, "trace": trace}

    def _binary_tree_search(self, question: str, max_depth: int) -> Tuple[str, List[dict]]:
        def recurse(q: str, depth: int, path: List[dict]) -> Any:
            if depth >= max_depth:
                return path

            chain = self.decomp_prompt | self.llm
            sub = chain.invoke({"question": q, "history": " ; ".join(p['subquery'] for p in path)}).content.strip()

            if sub.upper() == "TERMINATE":
                return path

            search_kwargs = {"k": 3}
            if self.dataset_ids:
                search_kwargs["filter"] = {"dataset_id": {"$in": self.dataset_ids}}
            
            docs = self.vectorstore.similarity_search(sub, **search_kwargs)
            context = "\n\n".join(d.page_content for d in docs)
            
            ans_chain = self.answer_prompt | self.llm
            intermediate = ans_chain.invoke({"query": sub, "context": context}).content
            
            new_step = {"subquery": sub, "retrieved": True, "answer": intermediate, "sources": docs}
            return recurse(sub, depth + 1, path + [new_step])

        final_path = recurse(question, 0, [])
        final_context = "\n\n".join(step['answer'] for step in final_path)
        
        final_answer_prompt = PromptTemplate.from_template(
            "Based on the following context, answer the main question.\nContext: {context}\nMain Question: {question}"
        )
        final_chain = final_answer_prompt | self.llm
        final_answer = final_chain.invoke({"context": final_context, "question": question}).content

        return final_answer, final_path
