from fastapi import APIRouter, Depends
from src.services.rag.rag_agent import RAGAgent
from src.api.schemas.rag import QueryRequest
from src.api.deps import get_agent

router = APIRouter(prefix="/rag")


@router.post("/query")
def rag_query(req: QueryRequest, agent: RAGAgent = Depends(get_agent)):
    if req.mode == "manual":
        return {"answer": agent.query(req.question)}
    elif req.mode == "lcel":
        return {"answer": agent.query_lcel(req.question)}
    return {"error": "Invalid mode. Use 'manual' or 'lcel'."}


@router.post("/reload")
def rag_reload(agent: RAGAgent = Depends(get_agent)):
    agent.load_vector_store()
    return {"status": "reloaded"}

