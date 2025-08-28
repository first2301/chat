"""API 의존성 주입(Dependencies).

- 요청 컨텍스트에서 `app.state.agent`를 안전하게 주입합니다.
- 에이전트 초기화 실패/지연 시 503(Service Unavailable)을 반환합니다.
"""

from typing import Optional
import os
from fastapi import Request, HTTPException, Header, Depends
from qdrant_client import QdrantClient
from backend.src.services.rag.rag_agent import RAGAgent
from backend.src.services.rag.config import Config


def get_agent(request: Request) -> RAGAgent:
    """앱 상태에서 RAG 에이전트를 주입합니다.

    Raises:
        HTTPException: 벡터스토어가 로드되지 않아 에이전트가 없는 경우(503)

    Returns:
        RAGAgent: 싱글톤 에이전트 인스턴스
    """
    agent: Optional[RAGAgent] = getattr(request.app.state, "agent", None)
    if agent is None:
        # 이론상 이제는 항상 존재해야 하지만, 방어적으로 503 유지
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent

