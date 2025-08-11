"""헬스 체크 엔드포인트.

- 서비스 상태 및 현재 설정 일부를 노출합니다.
- 운영 모니터링 및 헬스 프로브(liveness/readiness)에 사용합니다.
"""

from fastapi import APIRouter, Request
from rag_chatbot.backend.src.services.rag.config import Config

router = APIRouter()


@router.get("/healthz")
def healthz(request: Request):
    """서비스 상태를 반환합니다.

    Returns:\n
        dict: 벡터스토어 로드 여부 및 핵심 설정 값
    """
    agent_loaded = getattr(request.app.state, "agent", None) is not None
    return {
        "vector_store_loaded": agent_loaded,
        "embedding_model_name": Config.embedding_model_name,
        "ollama_model": Config.ollama_model_name,
        "ollama_base_url": getattr(Config, "ollama_base_url", None),
    }

