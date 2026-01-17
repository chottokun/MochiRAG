from langchain_core.prompts import PromptTemplate

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

DEFAULT_RAG_PROMPT = PromptTemplate(
    template=DEFAULT_RAG_PROMPT_TEMPLATE_STR,
    input_variables=["context", "question"]
)

CONVERSATIONAL_RAG_PROMPT_TEMPLATE_STR = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context and the chat history to answer the question.
If you don't know the answer, just say that you don't know.
Try to keep the answer concise and informative.
When you use information from the context, cite the source using the metadata (e.g., "According to [document_name from metadata], ...").

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:
"""

CONVERSATIONAL_RAG_PROMPT = PromptTemplate(
    template=CONVERSATIONAL_RAG_PROMPT_TEMPLATE_STR,
    input_variables=["chat_history", "context", "question"]
)

DEFAULT_STEP_BACK_TEMPLATE = """You are an expert at world knowledge. I am going to ask you a question. Your job is to formulate a single, more general question that captures the essence of the original question. Frame the question from the perspective of a historian or a researcher.
Original question: {question}
Step-back question:"""

DEFAULT_ACE_TOPIC_TEMPLATE = """Based on the following user question, identify the main topic or entity in one or two words.
Your answer should be concise and suitable for use as a database search key.
Examples:
- Question: "How does the ParentDocumentRetriever work in MochiRAG?" -> Answer: "ParentDocumentRetriever"
- Question: "Tell me about ensemble retrievers" -> Answer: "EnsembleRetriever"
- Question: "What are the key features?" -> Answer: "Features"

Original question: {question}
Topic:"""

DEFAULT_ACE_EVOLUTION_TEMPLATE = """You are an expert in synthesizing knowledge. Based on the user's question and the provided answer, formulate a single, concise, and reusable insight. This insight should be a piece of general knowledge that could help answer similar questions more effectively in the future.

Do not repeat the question or the answer. Focus on extracting the core principle or strategy.

User Question:
"{question}"

Provided Answer:
"{answer}"

Concise Insight:"""
