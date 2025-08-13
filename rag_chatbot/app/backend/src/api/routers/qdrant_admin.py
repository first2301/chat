"""Qdrant 관리자용 라우터.

목적:
- 관리자 전용으로 Qdrant 상태/컬렉션/포인트를 확인하고, 제한된 범위에서 탐색/집계를 제공합니다.

정책:
- 응답 크기 제한: limit 기본 20, 최대 200
- 벡터 응답은 기본 포함하지 않음(with_vector=false)
- 필터는 Qdrant Filter JSON 패스스루 방식을 권장
- 헤더 `X-Admin-Token`으로 간단 인증(환경변수 ADMIN_TOKEN)
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import QdrantClient

from backend.src.api.schemas.qdrant import (
	HealthResponse,
	CollectionsResponse,
	CollectionInfoResponse,
	ScrollRequest,
	ScrollResponse,
	CountRequest,
	CountResponse,
)
from backend.src.api.deps import get_qdrant_client, require_admin
from backend.src.services.rag.config import Config


router = APIRouter(prefix="/rag/admin/qdrant", tags=["qdrant-admin"], dependencies=[Depends(require_admin)])


@router.get("/health", response_model=HealthResponse, summary="Qdrant 연결 상태 확인", description="Qdrant 서버에 liveness를 요청하여 연결 가능 여부를 반환합니다.")
def health(client: QdrantClient = Depends(get_qdrant_client)):
	"""Qdrant 연결 상태를 반환합니다.

	Args:
		client: 의존성으로 주입되는 `QdrantClient`

	Returns:
		HealthResponse: 상태 문자열과 가능한 경우 버전
	"""
	try:
		res = client.get_liveness()
		return HealthResponse(status="ok", version=str(res.version) if hasattr(res, "version") else None)
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")


@router.get("/collections", response_model=CollectionsResponse, summary="컬렉션 목록", description="Qdrant에 존재하는 컬렉션들의 이름을 반환합니다.")
def list_collections(client: QdrantClient = Depends(get_qdrant_client)):
	"""컬렉션 목록을 반환합니다.

	Args:
		client: 의존성으로 주입되는 `QdrantClient`

	Returns:
		CollectionsResponse: 컬렉션 이름 리스트
	"""
	res = client.get_collections()
	return CollectionsResponse(collections=[c.name for c in res.collections])


@router.get("/collections/{name}/info", response_model=CollectionInfoResponse, summary="컬렉션 정보 요약", description="포인트 수, 벡터 차원/거리 메트릭, 상태를 요약합니다. 경로 파라미터 생략 시 기본 컬렉션 사용.")
def collection_info(name: Optional[str] = None, client: QdrantClient = Depends(get_qdrant_client)):
	"""컬렉션 정보 요약을 반환합니다.

	Args:
		name: 컬렉션 이름(미지정 시 `Config.qdrant_collection` 사용)
		client: 의존성으로 주입되는 `QdrantClient`

	Returns:
		CollectionInfoResponse
	"""
	name = name or Config.qdrant_collection
	info = client.get_collection(name)
	points = client.count(collection_name=name, exact=True).count
	vec_params = info.config.params.vectors if hasattr(info.config.params, "vectors") else None
	vector_size = getattr(vec_params, "size", None)
	distance = getattr(vec_params, "distance", None)
	status = getattr(info.status, "value", str(info.status)) if hasattr(info, "status") else None
	return CollectionInfoResponse(name=name, points_count=points, vector_size=vector_size, distance=str(distance) if distance else None, status=status)


@router.post("/collections/{name}/scroll", response_model=ScrollResponse, summary="포인트 스크롤 조회", description="Qdrant scroll API를 통해 필터/페이징으로 포인트를 조회합니다. `filter`에는 Qdrant Filter JSON을 그대로 전달하세요.")
def scroll_points(req: ScrollRequest, name: Optional[str] = None, client: QdrantClient = Depends(get_qdrant_client)):
	"""포인트를 스크롤 방식으로 조회합니다.

	Args:
		req: 스크롤 요청(필터/페이징/응답 옵션 포함)
		name: 컬렉션 이름(미지정 시 `Config.qdrant_collection` 사용)
		client: 의존성으로 주입되는 `QdrantClient`

	Notes:
		- offset/next_offset을 이용해 다음 페이지를 요청하세요.
		- with_vector 기본 False로 대용량 응답을 방지합니다.

	Returns:
		ScrollResponse: 포인트 배열과 다음 페이지 토큰
	"""
	name = name or Config.qdrant_collection
	res = client.scroll(collection_name=name, scroll_filter=req.filter, limit=req.limit, with_payload=req.with_payload, with_vectors=req.with_vector, offset=req.offset)
	points = [
		{"id": p.id, "payload": p.payload if req.with_payload else None, "vector": p.vector if req.with_vector else None}
		for p in res.points
	]
	return ScrollResponse(points=points, next_offset=res.next_page_offset)


@router.post("/collections/{name}/count", response_model=CountResponse, summary="포인트 카운트", description="필터 조건으로 포인트 개수를 집계합니다.")
def count_points(req: CountRequest, name: Optional[str] = None, client: QdrantClient = Depends(get_qdrant_client)):
	"""필터 조건으로 포인트 개수를 반환합니다.

	Args:
		req: 카운트 요청(필터 포함)
		name: 컬렉션 이름(미지정 시 `Config.qdrant_collection` 사용)
		client: 의존성으로 주입되는 `QdrantClient`

	Returns:
		CountResponse: 카운트 값
	"""
	name = name or Config.qdrant_collection
	count = client.count(collection_name=name, count_filter=req.filter, exact=False)
	return CountResponse(count=count.count)


