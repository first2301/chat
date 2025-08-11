"""RAG 시스템 설정 및 환경변수 로딩.

- 환경변수(.env)를 단일 지점에서 로드/관리합니다.
- 서비스/라우터는 본 설정만 참조하고 환경변수에 직접 접근하지 않습니다.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

# 환경파일 자동 탐색 유틸과 rag 루트 탐색 유틸
_explicit_env_file = os.getenv("ENV_FILE") or os.getenv("DOTENV_FILE")


def _find_rag_root(start: Path) -> Path:
    """현재 파일에서 상위로 올라가며 `rag_chatbot` 루트를 탐색합니다."""
    for parent in [start] + list(start.parents):
        if parent.name == "rag_chatbot":
            return parent
    # 실패 시 보수적으로 4단계 상위(…/rag_chatbot/backend/src/services/rag → rag_chatbot)
    try:
        return start.parents[4]
    except Exception:
        return start
"""
주의: Top-level에서 .env를 로드하지 않습니다. 앱 시작 시점에 Config.load_env()를 호출하세요.
"""


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

    # 내부 상태
    _env_loaded: bool = False
    loaded_env_path: Optional[str] = None
    _resolved_embedding_model_name: Optional[str] = None
    _resolved_data_dir: Optional[str] = None

    # 임베딩/환경 설정
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "embedding_models/BGE-m3-ko")
    embedding_device: Optional[str] = os.getenv("EMBEDDING_DEVICE")  # 'cpu' | 'cuda' | None(자동)

    # 문서 분할 설정
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # 검색 설정
    k: int = int(os.getenv("K", 20))

    # Ollama 설정
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL", "ko-llama-8B")

    # 벡터 저장소 설정 (FAISS 경로: 하위호환용; Qdrant 전환 시 미사용)
    vector_store_path: Optional[str] = os.getenv("VECTOR_STORE_PATH")

    # Qdrant 설정
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_collection")

    # 데이터 소스/자동 인덱싱 설정
    data_dir: str = os.getenv("DATA_DIR", "rag_chatbot/data")
    doc_globs_raw: str = os.getenv("DOC_GLOBS", "**/*.pdf,**/*.txt")
    on_missing_vector_store: str = os.getenv(
        "ON_MISSING_VECTOR_STORE", "auto_build"
    )  # auto_build | fail | llm_only

    @classmethod
    def load_env(cls, env_file: Optional[str] = None) -> None:
        """환경변수를 한 번만 로드합니다.

        우선순위: 인자 > ENV_FILE/DOTENV_FILE > rag_root/backend/{.env,.env.dev,.env.prod.dev} > CWD
        """
        if cls._env_loaded:
            return

        candidate: Optional[Path] = None
        if env_file:
            p = Path(env_file)
            candidate = p if p.exists() else None
        if candidate is None:
            if _explicit_env_file and Path(_explicit_env_file).exists():
                candidate = Path(_explicit_env_file)
        if candidate is None:
            here = Path(__file__).resolve()
            rag_root = _find_rag_root(here)
            backend_dir = rag_root / "backend"
            for name in [".env", ".env.dev", ".env.prod.dev"]:
                p = backend_dir / name
                if p.exists():
                    candidate = p
                    break
        if candidate is not None:
            load_dotenv(candidate)
            cls.loaded_env_path = str(candidate)
        else:
            load_dotenv()
            cls.loaded_env_path = None

        cls._env_loaded = True

    @classmethod
    def resolve_paths(cls) -> None:
        """상대경로를 rag_root 기준 절대경로로 해석하여 내부에 보관합니다."""
        here = Path(__file__).resolve()
        rag_root = _find_rag_root(here)
        backend_dir = rag_root / "backend"
        project_root = rag_root.parent

        # EMBEDDING_MODEL_NAME 해석
        name = cls.embedding_model_name
        p = Path(name)
        resolved: Optional[str] = None
        if p.is_absolute() and p.exists():
            resolved = str(p)
        else:
            for base in (backend_dir, rag_root, project_root):
                candidate = base / name
                if candidate.exists():
                    resolved = str(candidate)
                    break
        cls._resolved_embedding_model_name = resolved or name

        # DATA_DIR 해석
        dd = Path(cls.data_dir)
        if dd.is_absolute():
            cls._resolved_data_dir = str(dd)
        else:
            for base in (rag_root, backend_dir, project_root):
                candidate = base / cls.data_dir
                if candidate.exists():
                    cls._resolved_data_dir = str(candidate)
                    break
            else:
                cls._resolved_data_dir = str((rag_root / cls.data_dir).resolve())

    @classmethod
    def validate(cls) -> None:
        """필수값과 경로 유효성을 검증합니다."""
        missing = []
        for key in ["OLLAMA_BASE_URL", "OLLAMA_MODEL", "QDRANT_URL", "QDRANT_COLLECTION"]:
            if not os.getenv(key):
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # 로컬 경로로 해석된 임베딩이면 존재해야 함
        emb = cls._resolved_embedding_model_name or cls.embedding_model_name
        emb_path = Path(emb)
        looks_like_local = emb_path.is_absolute() or any(sep in emb for sep in ("/", "\\")) or emb.startswith(".")
        if looks_like_local and not emb_path.exists():
            raise ValueError(f"Embedding model path not found: {emb}")

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """임베딩 모델을 반환합니다. 디바이스(cpu/cuda) 자동 판단.

        - EMBEDDING_MODEL_NAME이 로컬 경로로 제공될 경우, 실행 위치와 무관하게
          프로젝트 기준 절대경로로 안전하게 해석합니다.
        """
        device = cls.embedding_device
        if device is None:
            try:
                import torch  # noqa: F401
                import torch.cuda  # noqa: F401
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # resolve_paths()가 사전 호출되면 해석된 경로 사용, 아니면 이전 방식으로 해석
        resolved_model_name = cls._resolved_embedding_model_name or cls.embedding_model_name

        return HuggingFaceEmbeddings(
            model_name=resolved_model_name,
            model_kwargs={"device": device},
        )

    @classmethod
    def get_data_dir(cls) -> str:
        """해석된 DATA_DIR를 반환합니다(미해석 시 원본)."""
        return cls._resolved_data_dir or cls.data_dir

    @classmethod
    def get_doc_globs(cls) -> list[str]:
        """문서 탐색 글롭 패턴 목록을 반환합니다."""
        patterns = [g.strip() for g in cls.doc_globs_raw.split(",") if g.strip()]
        return patterns or ["**/*.pdf", "**/*.txt"]

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