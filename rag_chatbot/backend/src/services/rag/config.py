"""RAG 시스템 설정 및 환경변수 로딩.

- 환경변수(.env)를 단일 지점에서 로드/관리합니다.
- 서비스/라우터는 본 설정만 참조하고 환경변수에 직접 접근하지 않습니다.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

load_dotenv()


class Config:
    """RAG 시스템 설정 클래스.

    Attributes:
        embedding_model_name: 임베딩 모델 이름 또는 로컬 경로
        embedding_device: 'cpu' | 'cuda' | None(자동 판단)
        chunk_size: 문서 분할 크기
        chunk_overlap: 문서 분할 중복 크기
        k: 검색 결과 상위 k개
        ollama_base_url: Ollama 서버 URL
        ollama_model_name: Ollama 모델 이름
        vector_store_path: 벡터 저장소 경로(필수 추천)
    """

    # 임베딩/환경 설정
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: Optional[str] = os.getenv("EMBEDDING_DEVICE")  # 'cpu' | 'cuda' | None(자동)

    # 문서 분할 설정
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # 검색 설정
    k: int = int(os.getenv("K", 20))

    # Ollama 설정
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL", "ko-llama-8B")

    # 벡터 저장소 설정 (.env에서만 관리; 기본값 없음)
    vector_store_path: Optional[str] = os.getenv("VECTOR_STORE_PATH")

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """임베딩 모델을 반환합니다. 디바이스(cpu/cuda) 자동 판단."""
        device = cls.embedding_device
        if device is None:
            try:
                import torch  # noqa: F401
                import torch.cuda  # noqa: F401
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        return HuggingFaceEmbeddings(
            model_name=cls.embedding_model_name,
            model_kwargs={"device": device},
        )

    @classmethod
    def set_embedding_model_name(cls, model_name: str):
        cls.embedding_model_name = model_name

    @classmethod
    def set_chunk_settings(cls, chunk_size: int, chunk_overlap: int):
        cls.chunk_size = chunk_size
        cls.chunk_overlap = chunk_overlap

    @classmethod
    def set_search_settings(cls, k: int):
        cls.k = k

    @classmethod
    def set_ollama_settings(cls, base_url: str, model_name: str):
        cls.ollama_base_url = base_url
        cls.ollama_model_name = model_name

    @classmethod
    def set_vector_store_path(cls, path: str):
        cls.vector_store_path = path