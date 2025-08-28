"""헬스 체크 엔드포인트.

- 서비스 상태 및 현재 설정 일부를 노출합니다.
- 운영 모니터링 및 헬스 프로브(liveness/readiness)에 사용합니다.
"""

from fastapi import APIRouter, Request
from qdrant_client import QdrantClient
from backend.src.services.rag.config import Config

router = APIRouter()


@router.get("/healthz")
def healthz(request: Request):
    """서비스 상태를 반환합니다.

    Returns:\n
        dict: 벡터스토어 로드 여부 및 핵심 설정 값
    """
    agent = getattr(request.app.state, "agent", None)
    agent_loaded = agent is not None
    # 체인 준비 상태 플래그 (agent가 있을 때만 평가)
    retriever_ready = bool(getattr(agent, "retriever", None)) if agent_loaded else None
    lcel_ready = bool(getattr(agent, "lcel_chain", None)) if agent_loaded else None
    qdrant_ok = None
    collection_exists = None
    try:
        client = QdrantClient(url=Config.qdrant_url, timeout=Config.qdrant_timeout)
        # 간단 연결 확인
        client.get_collections()
        qdrant_ok = True
        if agent_loaded:
            try:
                info = client.get_collection(Config.qdrant_collection)
                collection_exists = True if info else False
            except Exception:
                collection_exists = False
    except Exception:
        qdrant_ok = False

    return {
        "agent_initialized": agent_loaded,
        "retriever_ready": retriever_ready,
        "lcel_chain_ready": lcel_ready,
        "qdrant_reachable": qdrant_ok,
        "collection_exists": collection_exists,
        "embedding_model_name": Config.embedding_model_name,
        "ollama_model": Config.ollama_model_name,
        "ollama_base_url": getattr(Config, "ollama_base_url", None),
    }


@router.get("/")
async def root():
    return {"message": "FastAPI is running"}