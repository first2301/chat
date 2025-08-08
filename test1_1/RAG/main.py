from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from test1_1.RAG.src.services.rag.rag_agent import RAGAgent
from src.services.rag.config import Config

app = FastAPI(title="RAG Service")


@app.on_event("startup")
def on_startup():
    try:
        app.state.agent = RAGAgent()
        app.state.agent.load_vector_store()
    except Exception:
        app.state.agent = None


class QueryRequest(BaseModel):
    question: str
    mode: str = "lcel"  # 'manual' | 'lcel'


@app.get("/healthz")
def healthz(request: Request):
    agent: Optional[RAGAgent] = getattr(request.app.state, "agent", None)
    return {
        "vector_store_loaded": agent is not None,
        "embedding_model_name": Config.embedding_model_name,
        "ollama_model": getattr(Config, "ollama_model", getattr(Config, "ollama_model_name", None)),
        "ollama_base_url": Config.ollama_base_url,
    }


@app.post("/rag/query")
def rag_query(req: QueryRequest, request: Request):
    agent: Optional[RAGAgent] = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Vector store not loaded")
    try:
        if req.mode == "manual":
            return {"answer": agent.query(req.question)}
        elif req.mode == "lcel":
            return {"answer": agent.query_lcel(req.question)}
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'manual' or 'lcel'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/reload")
def rag_reload(request: Request):
    try:
        if getattr(request.app.state, "agent", None) is None:
            request.app.state.agent = RAGAgent()
        request.app.state.agent.load_vector_store()
        return {"status": "reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))