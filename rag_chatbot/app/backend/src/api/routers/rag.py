"""RAG 질의/관리 엔드포인트.

- query: RAG 질의 처리(수동/LCEL 체인 선택)
- reload: 벡터스토어 재로딩
"""

from fastapi import APIRouter, Depends
from backend.src.services.rag.rag_agent import RAGAgent
from backend.src.api.schemas.rag import (
    QueryRequest,
    IndexFilesRequest,
    IndexUrlsRequest,
    IndexTextRequest,
)
from backend.src.api.deps import get_agent

router = APIRouter(prefix="/rag")


@router.post("/query")
def rag_query(req: QueryRequest, agent: RAGAgent = Depends(get_agent)):
    """RAG 질의를 처리합니다.

    사용 방법:
        - mode="manual": 인덱스 문서를 검색해 컨텍스트를 주입하는 수동 체인 사용
        - mode="lcel": 리트리버-프롬프트-LLM을 LCEL로 연결한 체인 사용(기본)

    Args:
        req: 질문과 체인 모드("manual" | "lcel")

    Returns:
        dict: {"answer": str} — 생성된 답변 텍스트
    """
    if req.mode == "manual":
        return {"answer": agent.query(req.question)}
    elif req.mode == "lcel":
        return {"answer": agent.query_lcel(req.question)}
    return {"error": "Invalid mode. Use 'manual' or 'lcel'."}


@router.post("/reload")
def rag_reload(agent: RAGAgent = Depends(get_agent)):
    """벡터스토어를 다시 로드합니다.

    Returns:
        dict: {"status": "reloaded"}
    """
    agent.load_vector_store()
    return {"status": "reloaded"}

@router.post("/test_query/{question}")
def test_query(question: str, agent: RAGAgent = Depends(get_agent)):
    """검색 없이 LLM만 직접 호출해 간단 응답을 확인합니다.

    Returns:
        dict: {"answer": str}
    """
    # 벡터스토어가 없어도 LLM-only로 동작 가능
    return {"answer": agent.llm_query(question)}


@router.post("/index/files")
def index_files(req: IndexFilesRequest, agent: RAGAgent = Depends(get_agent)):
    """파일 경로 기반 인덱싱을 수행합니다.

    사용 방법:
        - 도커 환경: 컨테이너 내부 경로를 `paths`로 전달하세요(예: "/app/data/a.pdf").
        - `force_rebuild`는 현재 드롭 없이 업서트만 수행합니다.

    paths:
        - app/data/pdf
        - 필요에 따라 app/data 경로에 추가 파일 추가

    Returns:
        dict: {"status": "ok"}
    """
    if req.force_rebuild:
        # 간단한 재인덱싱: 현재는 drop 없이 추가 업서트로 단순화
        pass
    if req.paths:
        agent.add_files(req.paths)
    else:
        # 경로 미제공 시 DATA_DIR + DOC_GLOBS 스캔
        discovered = agent._discover_documents_from_data_dir()
        if discovered:
            agent.add_files(discovered)
    return {"status": "ok"}


@router.post("/index/urls")
def index_urls(req: IndexUrlsRequest, agent: RAGAgent = Depends(get_agent)):
    """URL 기반 인덱싱을 수행합니다.

    Args:
        req: 수집할 http(s) URL 목록

    urls:
        - https://www.google.com
        - https://www.naver.com
        - https://www.daum.net
        - 필요에 따라 app/data 경로에 추가 파일 추가

    Returns:
        dict: {"status": "ok"}
    """
    agent.add_urls(req.urls)
    return {"status": "ok"}


@router.post("/index/text")
def index_text(req: IndexTextRequest, agent: RAGAgent = Depends(get_agent)):
    """원문 텍스트 조각들을 인덱싱합니다.

    사용 방법:
        - `texts`: 원문 텍스트 리스트. 각 항목이 분할→임베딩→업서트됩니다.
        - `source`: 각 텍스트의 메타데이터 `source` 라벨(예: "inline", "ticket-123").

    texts:
        - 원문 텍스트 리스트
        - 필요에 따라 app/data 경로에 추가 파일 추가
        - 예) "회사 소개", "회사 연혁", "회사 문화"

    Returns:
        dict: {"status": "ok" | "skipped"}
    """
    from langchain_core.documents import Document
    docs = [Document(page_content=t, metadata={"source": req.source or "inline"}) for t in req.texts]
    split_docs = agent.ingestion.split(docs)
    if hasattr(agent.vector_store, "add_documents"):
        agent.vector_store.add_documents(split_docs)
        agent._initialize_rag_chain()
        return {"status": "ok"}
    return {"status": "skipped"}