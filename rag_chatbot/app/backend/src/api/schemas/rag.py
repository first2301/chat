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
    """파일 인덱싱 요청 스키마.

    사용 방법(출처 입력 가이드):
    - paths: 백엔드 프로세스가 접근 가능한 파일 경로 목록을 입력합니다.
      도커에서 실행 중이면 "컨테이너 내부 경로"를 사용해야 합니다.
      예) "/app/data/report.pdf", "/app/data/docs/readme.txt"
    - globs: 글롭 패턴 목록(예: "**/*.pdf", "docs/**/*.txt").
      현재 라우터 구현에서는 이 필드를 사용하지 않고, `Config.DOC_GLOBS`(환경변수 DOC_GLOBS)를 사용합니다.
      필요 시 라우터 확장 후 사용하세요.
    - force_rebuild: 재인덱싱 플래그(현재는 드롭 없이 추가 업서트만 수행).
    """
    paths: Optional[List[str]] = None
    globs: Optional[List[str]] = None
    force_rebuild: bool = False


class IndexUrlsRequest(BaseModel):
    """URL 인덱싱 요청 스키마.

    사용 방법(출처 입력 가이드):
    - urls: http(s) 스킴의 웹 문서 URL 목록을 입력합니다.
      예) "https://example.com/page", "https://docs.example.com/guide.html"
      주의: robots/인증/차단 정책에 따라 수집이 제한될 수 있습니다.
    """
    urls: List[str]


class IndexTextRequest(BaseModel):
    """텍스트 인덱싱 요청 스키마.

    사용 방법(출처 입력 가이드):
    - texts: 원문 텍스트 조각들의 리스트입니다. 각 항목이 분할 후 임베딩되어 인덱싱됩니다.
    - source: 메타데이터 "source"에 기록될 출처 라벨(선택). 예) "inline", "ticket-123", "manual-upload".
      지정하지 않으면 기본값으로 "inline"이 사용됩니다.
    """
    texts: List[str]
    source: Optional[str] = None

