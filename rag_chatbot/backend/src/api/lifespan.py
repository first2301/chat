"""애플리케이션 수명(lifespan) 관리.

- 앱 시작 시 `RAGAgent`를 초기화하고 벡터스토어를 로드합니다.
- 초기 인덱스가 없거나 로드 실패 시 `app.state.agent = None`로 설정합니다.
- 종료 시 별도 정리가 필요하면 이 컨텍스트의 종료 블록에 추가합니다.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from rag_chatbot.backend.src.services.rag.rag_agent import RAGAgent


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """FastAPI lifespan 컨텍스트.

    앱 시작 시 RAG 에이전트를 생성하고 가능한 경우 벡터스토어를 로드합니다.
    실패하면 `app.state.agent`는 None으로 설정되어, 라우터에서 503을 반환하도록 합니다.
    """
    try:
        agent = RAGAgent()
        agent.load_vector_store()
        app.state.agent = agent
    except Exception:
        app.state.agent = None
    yield
    # 종료 시 정리 필요하면 여기에 추가

