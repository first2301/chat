"""RAG 질의/관리 엔드포인트.

- query: RAG 질의 처리(수동/LCEL 체인 선택)
- reload: 벡터스토어 재로딩
"""

from fastapi import APIRouter, Depends
from rag_chatbot.backend.src.services.rag.rag_agent import RAGAgent
from rag_chatbot.backend.src.api.schemas.rag import QueryRequest, IndexFilesRequest, IndexUrlsRequest, IndexTextRequest
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

@router.post("/test_query/{question}")
def test_query(question: str, agent: RAGAgent = Depends(get_agent)):
    """테스트 질의를 처리합니다."""
    # 벡터스토어가 없어도 LLM-only로 동작 가능
    return {"answer": agent.llm_query(question)}


@router.post("/index/files")
def index_files(req: IndexFilesRequest, agent: RAGAgent = Depends(get_agent)):
    """파일 경로 기반 인덱싱을 수행합니다."""
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
    agent.add_urls(req.urls)
    return {"status": "ok"}


@router.post("/index/text")
def index_text(req: IndexTextRequest, agent: RAGAgent = Depends(get_agent)):
    # 간단 구현: 텍스트를 임시 파일로 취급하지 않고 Document로 직접 처리하고 싶지만,
    # 현재 RAGAgent는 파일/URL 기반. 간단히 메모리 로딩 경로를 재사용하려면 내부 확장이 필요.
    # 여기서는 최소 구현으로 텍스트를 일시 파일로 처리하는 대신, 직접 처리 경로는 TODO로 남김.
    from langchain_core.documents import Document
    docs = [Document(page_content=t, metadata={"source": req.source or "inline"}) for t in req.texts]
    # 분할 후 업서트
    split_docs = agent._split_documents(docs)
    # Qdrant 래퍼는 add_documents 지원
    if hasattr(agent.vector_store, "add_documents"):
        agent.vector_store.add_documents(split_docs)
        agent._initialize_rag_chain()
        return {"status": "ok"}
    return {"status": "skipped"}