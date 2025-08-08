from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.services.rag.rag_agent import RAGAgent


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    try:
        agent = RAGAgent()
        agent.load_vector_store()
        app.state.agent = agent
    except Exception:
        app.state.agent = None
    yield
    # 종료 시 정리 필요하면 여기에 추가

