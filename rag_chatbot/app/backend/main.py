"""FastAPI 애플리케이션 엔트리 포인트.

- 앱 팩토리(`create_app`)에서 라우터를 연결하고 lifespan 컨텍스트를 설정합니다.
- 실행 시 `app` 객체를 사용하여 Uvicorn에서 서비스를 기동합니다.
- CORS는 배포 환경에서 제한적으로 설정하는 것을 권장합니다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.api.routers import health as health_router
from backend.src.api.routers import rag as rag_router
from backend.src.api.lifespan import lifespan_context


def create_app() -> FastAPI:
    """FastAPI 애플리케이션을 생성합니다.

    Returns:
        FastAPI: RAG 서비스 앱 인스턴스. 헬스 및 RAG 라우터가 포함되며
        lifespan 컨텍스트를 통해 `app.state.agent`를 수명 주기 동안 관리합니다.
    """
    app = FastAPI(title="RAG Service", lifespan=lifespan_context)
    # CORS 설정: 프론트엔드 오리진을 환경변수 또는 널널한 기본값으로 허용
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 배포 시 특정 오리진으로 제한 권장
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router.router)
    app.include_router(rag_router.router)
    return app


app = create_app()
