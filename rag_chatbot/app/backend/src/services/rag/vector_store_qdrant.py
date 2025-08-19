"""Qdrant 벡터스토어 관리 (langchain-qdrant 사용)."""

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
        qdrant_api_key: Qdrant API 키(미지정 시 `Config.qdrant_api_key`).

    사용 방법:
        >>> mgr = VectorStoreManagerQdrant(embeddings)
        >>> retriever = mgr.get_retriever(k=5)
        >>> mgr.add_documents(split_docs)

    특징:
    - 컬렉션이 없으면 임베딩 차원을 추론하여 자동 생성합니다.
    - `vs` 속성에 LangChain `QdrantVectorStore` 인스턴스를 노출합니다.
    """

    def __init__(
            self,
            embeddings: HuggingFaceEmbeddings,
            collection: Optional[str] = None,
            qdrant_url: Optional[str] = None,
            qdrant_api_key: Optional[str] = None,
    ):
        self.client = QdrantClient(
            url=qdrant_url if qdrant_url is not None else Config.qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key is not None else Config.qdrant_api_key,
        )
        self.collection = collection or Config.qdrant_collection
        self.embeddings = embeddings
        self._ensure_collection()
        self.vs = QdrantVectorStore(client=self.client, collection_name=self.collection, embedding=self.embeddings)

    def _ensure_collection(self) -> None:
        """컬렉션 존재 보장. 없으면 생성합니다.

        동작:
        - 존재 확인 실패 시, 임베딩 차원을 한 번 계산하여 벡터 크기를 추론하고 생성합니다.

        Raises:
            RuntimeError: 임베딩 차원 추론 실패 시
        """
        try:
            self.client.get_collection(self.collection)
            return
        except Exception:
            pass
        # 임베딩 차원 계산 후 컬렉션 생성
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
        """
        self.vs.add_documents(documents)


