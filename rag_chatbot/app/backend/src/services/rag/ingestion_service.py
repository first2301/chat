"""인제스천 서비스: 파일 스캔 + 로딩 + 분할 통합."""

from __future__ import annotations

from pathlib import Path
import os
from typing import List

from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.src.services.rag.config import Config


class IngestionService:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""],
        )
        # 프로젝트 루트 디렉토리 계산 ("rag_chatbot" 또는 "backend" 어느 쪽이든 인식)
        here = Path(__file__).resolve()
        self._rag_root = here
        for parent in [here] + list(here.parents):
            if parent.name in ("rag_chatbot", "backend"):
                self._rag_root = parent
                break

    def discover_files(self, data_dir: str | None = None, patterns: List[str] | None = None) -> List[str]:
        """
        데이터 디렉토리에서 파일을 검색하고 파일 경로를 반환합니다.

        Args:
            data_dir: 데이터 디렉토리 경로
            patterns: 파일 패턴 목록

        Returns:
        """
        base = Path(data_dir or Config.get_data_dir())
        patterns = patterns or Config.get_doc_globs()
        results: List[str] = []
        for pattern in patterns:
            for p in base.glob(pattern):
                if p.is_file():
                    try:
                        p.resolve().relative_to(self._rag_root)
                    except Exception:
                        # rag_chatbot 외부는 무시
                        continue
                    results.append(str(p.resolve()))
        return sorted(list(dict.fromkeys(results)))

    def load_files(self, paths: List[str]):
        from langchain_core.documents import Document

        documents = []
        file_list: List[str] = []

        # 디렉터리 경로가 섞여 들어올 수 있으므로 파일 목록으로 확장
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                # 지정 디렉터리 하위에서 허용 글롭만 재귀 탐색
                for pattern in Config.get_doc_globs():
                    for f in p.glob(pattern):
                        if f.is_file():
                            file_list.append(str(f))
            elif p.is_file():
                file_list.append(str(p))
            else:
                # 존재하지 않거나 접근 불가 경로는 건너뜀
                continue

        for path in sorted(list(dict.fromkeys(file_list))):
            if path.lower().endswith(".pdf"):
                loader = PyMuPDFLoader(path)
                documents.extend(loader.load())
            else:
                with open(path, "r", encoding="utf-8") as f:
                    documents.append(Document(page_content=f.read(), metadata={"source": path}))
        return documents

    def load_urls(self, urls: List[str]):
        loader = WebBaseLoader(
            web_paths=urls,
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
            },
        )
        return loader.load()

    def split(self, documents: List):
        return self._splitter.split_documents(documents)

    def documents_from_texts(self, texts: List[str], source: str | None = None):
        from langchain_core.documents import Document

        return [Document(page_content=t, metadata={"source": source or "inline"}) for t in texts]


