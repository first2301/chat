"""RAG 질의/관리 엔드포인트.

- query: RAG 질의 처리(수동/LCEL 체인 선택)
- reload: 벡터스토어 재로딩
"""

from fastapi import APIRouter, Depends
from rag_chatbot.backend.src.services.rag.rag_agent import RAGAgent
from rag_chatbot.backend.src.api.schemas.rag import QueryRequest
from rag_chatbot.backend.src.api.deps import get_agent

router = APIRouter(prefix="/rag")


@router.post("/query")
def rag_query(req: QueryRequest, agent: RAGAgent = Depends(get_agent)):
    """RAG 질의를 처리합니다.

    Args:
        req (QueryRequest): 질문과 체인 모드("manual" | "lcel")

    Returns:
        dict: {"answer": str}
    """
    if req.mode == "manual":
        return {"answer": agent.query(req.question)}
    elif req.mode == "lcel":
        return {"answer": agent.query_lcel(req.question)}
    return {"error": "Invalid mode. Use 'manual' or 'lcel'."}


@router.post("/reload")
def rag_reload(agent: RAGAgent = Depends(get_agent)):
    """벡터스토어를 다시 로드합니다."""
    agent.load_vector_store()
    return {"status": "reloaded"}

