from fastapi import FastAPI

from src.api.routers import health as health_router
from src.api.routers import rag as rag_router
from src.api.lifespan import lifespan_context


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Service", lifespan=lifespan_context)
    app.include_router(health_router.router)
    app.include_router(rag_router.router)
    return app


app = create_app()

