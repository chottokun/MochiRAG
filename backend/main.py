from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from .database import engine
from . import models
from .routers import auth, users, datasets, documents, chat
from core.vector_store_manager import vector_store_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create all database tables on startup
# In a production environment, it's better to use migrations (e.g., Alembic)
models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, initialize the ChromaDB client
    logger.info("Starting up: Initializing Vector Store Client")
    vector_store_manager.initialize_client()
    yield
    # On shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="MochiRAG API",
    description="Multi-tenant RAG application API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Global Exception Handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# --- Include Routers ---

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(datasets.router)
app.include_router(documents.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Welcome to MochiRAG API"}
