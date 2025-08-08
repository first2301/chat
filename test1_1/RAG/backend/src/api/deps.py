from typing import Optional
from fastapi import Request, HTTPException
from src.services.rag.rag_agent import RAGAgent


def get_agent(request: Request) -> RAGAgent:
    agent: Optional[RAGAgent] = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Vector store not loaded")
    return agent

