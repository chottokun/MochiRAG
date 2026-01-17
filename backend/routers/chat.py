from fastapi import APIRouter, Depends, BackgroundTasks
from .. import schemas, models
from ..dependencies import get_current_user
from core.rag_chain_service import rag_chain_service
from core.config_manager import config_manager
from core.context_evolution_service import context_evolution_service

router = APIRouter(prefix="/chat", tags=["chat"])

@router.get("/strategies/")
def get_available_rag_strategies():
    """Returns a list of available RAG strategies from the config."""
    retriever_configs = config_manager.config.retrievers
    return {"strategies": list(retriever_configs.keys())}

@router.post("/query/", response_model=schemas.QueryResponse)
def query_rag_chain(
    query: schemas.QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(get_current_user)
):
    response = rag_chain_service.get_rag_response(query, user_id=current_user.id)

    # If the ACE strategy was used, it will have returned a topic.
    # Use this topic to evolve the context in the background.
    if response.topic:
        background_tasks.add_task(
            context_evolution_service.evolve_context_from_interaction,
            user_id=current_user.id,
            question=query.query,
            answer=response.answer,
            topic=response.topic
        )

    return response
