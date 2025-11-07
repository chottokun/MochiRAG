import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend import crud
from backend.database import SessionLocal
from .llm_manager import llm_manager
from .config_manager import config_manager

logger = logging.getLogger(__name__)

DEFAULT_EVOLUTION_TEMPLATE = """You are an expert in synthesizing knowledge. Based on the user's question and the provided answer, formulate a single, concise, and reusable insight. This insight should be a piece of general knowledge that could help answer similar questions more effectively in the future.

Do not repeat the question or the answer. Focus on extracting the core principle or strategy.

User Question:
"{question}"

Provided Answer:
"{answer}"

Concise Insight:"""


class ContextEvolutionService:
    def __init__(self):
        self.llm = llm_manager.get_llm()
        self._setup_chains()

    def _setup_chains(self):
        # Chain for generating the evolved context
        template = config_manager.get_prompt("ace_evolution", default=DEFAULT_EVOLUTION_TEMPLATE)
        evolution_prompt = PromptTemplate.from_template(template)
        self.evolution_chain = evolution_prompt | self.llm | StrOutputParser()

    def evolve_context_from_interaction(self, user_id: int, question: str, answer: str, topic: str):
        """
        Generates and saves an evolved context from a Q&A interaction.
        """
        logger.info(f"Starting context evolution for user {user_id} on topic '{topic}'")

        if not all([question, answer, topic]):
            logger.warning("Missing question, answer, or topic. Aborting context evolution.")
            return

        # Generate the new insight (the evolved context)
        evolved_content = self.evolution_chain.invoke({"question": question, "answer": answer})
        if not evolved_content:
            logger.warning("Context evolution failed to generate content. Aborting.")
            return

        logger.info(f"Generated new insight for topic '{topic}': '{evolved_content[:100]}...'")

        # 3. Save the new context to the database
        db = SessionLocal()
        try:
            crud.create_evolved_context(
                db=db,
                owner_id=user_id,
                content=evolved_content,
                topic=topic.strip()
            )
            logger.info(f"Successfully saved new evolved context for topic '{topic}' to the database.")
        except Exception as e:
            logger.error(f"Failed to save evolved context to the database: {e}")
        finally:
            db.close()

# Create a single, globally accessible instance
context_evolution_service = ContextEvolutionService()
