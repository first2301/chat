from fastapi import APIRouter
from src.services.rag.config import Config

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {
        "embedding_model_name": getattr(Config, "embedding_model_name", None),
        "ollama_model": getattr(Config, "ollama_model", getattr(Config, "ollama_model_name", None)),
        "ollama_base_url": getattr(Config, "ollama_base_url", None),
    }

