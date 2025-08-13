"""Qdrant 관리자용 요청/응답 스키마.

관리 기능은 응답 크기를 제한하고, 필터는 Qdrant의 Filter JSON을 그대로 전달하는 방식을 권장합니다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	"""Qdrant 헬스 응답.

	Attributes:
		status: "ok" 또는 오류 상황에 따른 상태 문자열
		version: Qdrant 버전 문자열(가능한 경우)
	"""

	status: str
	version: Optional[str] = None


class CollectionsResponse(BaseModel):
	"""컬렉션 목록 응답.

	Attributes:
		collections: 컬렉션 이름 리스트
	"""

	collections: List[str]


class CollectionInfoResponse(BaseModel):
	"""컬렉션 정보 요약 응답.

	Attributes:
		name: 컬렉션 이름
		points_count: 포인트 총 개수(정확 카운트)
		vector_size: 벡터 차원 수
		distance: 거리 메트릭(e.g., COSINE)
		status: 컬렉션 상태 문자열
	"""

	name: str
	points_count: Optional[int] = None
	vector_size: Optional[int] = None
	distance: Optional[str] = None
	status: Optional[str] = None


class ScrollRequest(BaseModel):
	"""포인트 스크롤 요청.

	Notes:
		- filter: Qdrant Filter JSON을 그대로 전달합니다.
		- limit: 기본 20, 최대 200으로 대용량 응답을 방지합니다.
		- with_vector: 기본 False(필요한 경우에만 True로 설정).
	"""

	filter: Optional[Dict[str, Any]] = None
	limit: int = Field(default=20, ge=1, le=200)
	offset: Optional[str] = None
	with_payload: bool = True
	with_vector: bool = False


class PointPayload(BaseModel):
	"""단일 포인트 페이로드.

	Attributes:
		id: 포인트 식별자
		payload: 메타데이터 딕셔너리(옵션)
		vector: 벡터(옵션, 요청 시에만 포함)
	"""

	id: Any
	payload: Optional[Dict[str, Any]] = None
	vector: Optional[Any] = None


class ScrollResponse(BaseModel):
	"""스크롤 결과 응답.

	Attributes:
		points: 반환된 포인트 배열
		next_offset: 다음 페이지 토큰(None이면 마지막 페이지)
	"""

	points: List[PointPayload]
	next_offset: Optional[str] = None


class CountRequest(BaseModel):
	"""포인트 카운트 요청.

	Attributes:
		filter: Qdrant Filter JSON
	"""

	filter: Optional[Dict[str, Any]] = None


class CountResponse(BaseModel):
	"""포인트 카운트 응답.

	Attributes:
		count: 조건에 부합하는 포인트 개수
	"""

	count: int


class SamplesResponse(BaseModel):
	"""샘플 포인트 응답."""

	samples: List[PointPayload]


class PayloadKeysResponse(BaseModel):
	"""페이로드 키 목록과 예시 응답."""

	keys: List[str]
	examples: Dict[str, List[Any]]


