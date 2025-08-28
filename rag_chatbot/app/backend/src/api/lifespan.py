"""애플리케이션 수명(lifespan) 관리.

- 앱 시작 시 환경 로드/경로 해석/검증을 수행하고 `RAGAgent`를 초기화합니다.
- 초기 인덱스가 없거나 로드 실패 시에도 앱은 기동하며, 의존성에서 503이 반환될 수 있습니다.
- 종료 시 별도 정리가 필요하면 종료 블록에 추가합니다.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from backend.src.services.rag.rag_agent import RAGAgent
from backend.src.services.rag.config import Config


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """FastAPI lifespan 컨텍스트.

    동작 개요:
    - 환경변수 로드(`.env/.env.dev/.env.prod.dev` 탐색) 및 경로 해석/검증
    - `RAGAgent` 생성(Qdrant 우선, 정책에 따라 자동 인덱싱 가능)
    - 앱 상태에 에이전트 바인딩(`app.state.agent`)

    예외 처리:
    - 초기화 실패 시에도 앱은 기동하며, 의존성에서 503을 반환할 수 있습니다.
    """
    # 설정 로드/검증 및 에이전트 초기화를 보호 구간으로 묶어, 실패해도 앱은 기동
    Config.load_env()
    Config.resolve_paths()
    Config.validate()
    try:
        # Config.load_env()
        # Config.resolve_paths()
        # Config.validate()
        agent = RAGAgent()
    except Exception as e:
        logging.exception("Initialization during lifespan failed: %s", e)
        agent = None
    app.state.agent = agent
    yield
    # 종료 시 정리 필요하면 여기에 추가

