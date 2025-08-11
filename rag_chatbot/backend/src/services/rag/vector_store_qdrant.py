"""Qdrant 벡터스토어 관리 (langchain-qdrant 사용)."""

from __future__ import annotations

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from rag_chatbot.backend.src.services.rag.config import Config


class VectorStoreManagerQdrant:
    def __init__(self, embeddings: HuggingFaceEmbeddings, collection: Optional[str] = None):
        self.client = QdrantClient(url=Config.qdrant_url, api_key=Config.qdrant_api_key)
        self.collection = collection or Config.qdrant_collection
        self.embeddings = embeddings
        self._ensure_collection()
        self.vs = QdrantVectorStore(client=self.client, collection_name=self.collection, embedding=self.embeddings)

    def _ensure_collection(self) -> None:
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
        return self.vs.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: List):
        self.vs.add_documents(documents)


