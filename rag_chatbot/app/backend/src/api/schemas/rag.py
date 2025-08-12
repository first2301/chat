"""API 요청/응답 스키마 정의."""

from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    """RAG 질의 요청 스키마.

    Attributes:
        question: 사용자의 질문 텍스트
        mode: 체인 모드 선택("manual" | "lcel"), 기본값은 "lcel"
    """
    question: str
    mode: str = "lcel"  # 'manual' | 'lcel'


class IndexFilesRequest(BaseModel):
    paths: Optional[List[str]] = None
    globs: Optional[List[str]] = None
    force_rebuild: bool = False


class IndexUrlsRequest(BaseModel):
    urls: List[str]


class IndexTextRequest(BaseModel):
    texts: List[str]
    source: Optional[str] = None

