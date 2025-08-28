"""Qdrant 벡터스토어 관리 (langchain-qdrant 사용).

특징/정책:
- 컬렉션 자동 보장: 컬렉션이 없으면 임베딩 차원을 단 한 번 추론하여 생성합니다.
- 연결 안정성: Qdrant 부팅 지연을 고려하여 제한적 재시도를 수행합니다.
- 거리함수: COSINE 고정(문장 임베딩에 적합). 필요 시 확장 가능합니다.

주의사항:
- 임베딩 모델 변경(차원 변경) 시 기존 컬렉션과 차원이 불일치하면 에러가 발생합니다.
  이 경우 새로운 컬렉션명을 사용하거나 기존 컬렉션을 삭제 후 재생성하세요.
"""

from __future__ import annotations

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Config


class VectorStoreManagerQdrant:
    """Qdrant 벡터스토어 매니저.

    Args:
        embeddings: 임베딩 계산기(`HuggingFaceEmbeddings`).
        collection: 사용할 Qdrant 컬렉션 이름(미지정 시 `Config.qdrant_collection`).
        qdrant_url: Qdrant 서버 URL(미지정 시 `Config.qdrant_url`).
        # qdrant_api_key: Qdrant API 키(미지정 시 `Config.qdrant_api_key`).

    사용 방법:
        >>> mgr = VectorStoreManagerQdrant(embeddings)
        >>> retriever = mgr.get_retriever(k=5)
        >>> mgr.add_documents(split_docs)

    특징:
    - 컬렉션 미존재 시 임베딩 차원을 추론하여 자동 생성합니다.
    - `vs` 속성에 LangChain `QdrantVectorStore` 인스턴스를 노출합니다.
    - 거리함수는 COSINE, 벡터 크기는 추론된 차원을 사용합니다.
    """

    def __init__(
            self,
            embeddings: HuggingFaceEmbeddings,
            collection: Optional[str] = None,
            qdrant_url: Optional[str] = None,
            # qdrant_api_key: Optional[str] = None,
    ):
        # 연결 지연을 고려해 타임아웃/재시도 포함 클라이언트 생성
        self.client = QdrantClient(
            url=qdrant_url if qdrant_url is not None else Config.qdrant_url,
            # api_key=qdrant_api_key if qdrant_api_key is not None else Config.qdrant_api_key,
            timeout=Config.qdrant_timeout,
        )
        self.collection = collection or Config.qdrant_collection
        self.embeddings = embeddings
        self._ensure_collection()
        self.vs = QdrantVectorStore(client=self.client, collection_name=self.collection, embedding=self.embeddings)

    def _ensure_collection(self) -> None:
        """컬렉션 존재를 보장합니다. 없거나 준비되지 않았으면 생성까지 수행합니다.

        프로세스:
        1) Qdrant 부팅 지연을 고려해 `get_collection`을 제한적으로 재시도합니다.
        2) 컬렉션이 없으면 임베딩 차원을 단 한 번 계산해 벡터 크기를 추론합니다.
        3) 추론된 차원과 COSINE 거리로 컬렉션을 생성합니다.

        Raises:
            RuntimeError: 임베딩 차원을 추론하지 못한 경우
        """
        # Qdrant가 부팅 중일 수 있으므로 제한적 재시도
        import time
        retries = max(0, Config.qdrant_connect_retries)
        interval = max(0.1, Config.qdrant_connect_retry_interval)
        for attempt in range(retries + 1):
            try:
                self.client.get_collection(self.collection)
                return
            except Exception:
                if attempt < retries:
                    time.sleep(interval)
                else:
                    break
        # (컬렉션 미존재 시) 임베딩 차원 추론 후 컬렉션 생성
        try:
            dim = len(self.embeddings.embed_query("dimension probe"))
        except Exception as e:
            raise RuntimeError(f"Failed to infer embedding dimension: {e}")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def get_retriever(self, k: int):
        """리트리버를 반환합니다.

        Args:
            k: 유사도 검색 상위 개수

        Returns:
            Retriever: LangChain 리트리버
        """
        return self.vs.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: List):
        """분할된 문서 리스트를 업서트합니다.

        Args:
            documents: LangChain `Document` 리스트
        Note:
            - Qdrant는 업서트 시 기존 포인트와 중복될 수 있습니다. 중복 방지 필요 시 외부에서 ID를 관리하세요.
        """
        self.vs.add_documents(documents)


