from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class QueryResponse(BaseModel):
    role: str = "assistant"
    answer: str

class QueryRequest(BaseModel):
    role: str
    content: str
    # question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발단계는 * / 운영은 특정 도메인만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/query", response_model=QueryResponse)
def rag_query_get(req: QueryRequest):
    """RAG 질의를 처리합니다.

    Args:
        question: 질문 텍스트(경로 파라미터)
        agent: 의존성 주입된 `RAGAgent`

    Returns:
        QueryResponse: {"answer": str}
    """
    answer = req.content
    return QueryResponse(answer=answer)