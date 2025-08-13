"""API 의존성 주입(Dependencies).

- 요청 컨텍스트에서 `app.state.agent`를 안전하게 주입합니다.
- 벡터스토어가 아직 로드되지 않았다면 503을 반환합니다.
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


def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 의존성.

    - `Config.qdrant_url`/`Config.qdrant_api_key`를 사용해 클라이언트를 생성합니다.
    - 연결 실패 시 예외를 전파하여 5xx로 처리되도록 합니다.
    """
    return QdrantClient(url=Config.qdrant_url, api_key=Config.qdrant_api_key)


def require_admin(x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    """간단한 관리자 토큰 검사.

    - 요청 헤더 `X-Admin-Token`을 환경변수 `ADMIN_TOKEN`과 비교합니다.
    - `ADMIN_TOKEN`이 비어 있으면 보호가 비활성화되며, 로컬 개발 용도로만 권장합니다.
    - 토큰이 일치하지 않으면 401 Unauthorized를 반환합니다.
    """
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        # 관리자 보호가 비활성화된 환경
        return
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return

