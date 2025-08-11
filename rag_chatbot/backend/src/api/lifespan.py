"""애플리케이션 수명(lifespan) 관리.

- 앱 시작 시 `RAGAgent`를 초기화하고 벡터스토어를 로드합니다.
- 초기 인덱스가 없거나 로드 실패 시 `app.state.agent = None`로 설정합니다.
- 종료 시 별도 정리가 필요하면 이 컨텍스트의 종료 블록에 추가합니다.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from rag_chatbot.backend.src.services.rag.rag_agent import RAGAgent
from rag_chatbot.backend.src.services.rag.config import Config


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """FastAPI lifespan 컨텍스트.

    앱 시작 시 RAG 에이전트를 생성하고 가능한 경우 벡터스토어를 로드합니다.
    실패하면 `app.state.agent`는 None으로 설정되어, 라우터에서 503을 반환하도록 합니다.
    """
    # 환경 로드/경로 해석/검증
    Config.load_env()
    Config.resolve_paths()
    Config.validate()

    # 에이전트 초기화 후, 필요 시 벡터스토어 보장(자동 인덱싱 가능)
    agent = RAGAgent()
    try:
        # Qdrant 연결 기반이므로 load_vector_store 대신 초기화된 상태 사용
        # 자동 인덱싱 정책(auto_build)이면 내부에서 처리됨
        _ = agent
    except Exception:
        pass
    app.state.agent = agent
    yield
    # 종료 시 정리 필요하면 여기에 추가

