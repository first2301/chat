"""인제스천 서비스: 파일 스캔 + 로딩 + 분할 통합."""

from __future__ import annotations

from pathlib import Path
import os
from typing import List
import re

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
        self._url_line_regex = re.compile(r"^https?://\S+$", re.IGNORECASE)

    def discover_files(self, data_dir: str | None = None, patterns: List[str] | None = None) -> List[str]:
        """
        데이터 디렉토리에서 파일을 검색하고 파일 경로를 반환합니다.

        Args:
            data_dir: 데이터 디렉토리 경로
            patterns: 파일 패턴 목록

        Returns:
        """
        # DATA_DIR 내부에서만 스캔하도록 기준 디렉터리를 엄격히 고정합니다.
        base = Path(data_dir or Config.get_data_dir()).resolve()
        patterns = patterns or Config.get_doc_globs()
        results: List[str] = []
        for pattern in patterns:
            for p in base.glob(pattern):
                if p.is_file():
                    # 안전 장치: 반드시 DATA_DIR 하위인 경우에만 포함
                    try:
                        p.resolve().relative_to(base)
                    except Exception:
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
            elif path.lower().endswith(".txt"):
                # URL 목록 파일 규칙: 디렉터리명이 urls 또는 파일명이 *.url.txt / *.urls.txt
                p_obj = Path(path)
                is_url_dir = any(seg.lower() == "urls" for seg in p_obj.parts)
                is_url_suffix = p_obj.name.lower().endswith((".url.txt", ".urls.txt"))
                if is_url_dir or is_url_suffix:
                    # 줄 단위로 URL만 추출하여 웹 수집으로 전환
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        url_lines = [ln.strip() for ln in f.readlines()]
                    urls = [ln for ln in url_lines if ln and self._url_line_regex.match(ln)]
                    if urls:
                        # 중복 제거 및 순서 안정화
                        urls = sorted(list(dict.fromkeys(urls)))
                        documents.extend(self.load_urls(urls))
                        continue
                    # URL이 하나도 없으면 일반 텍스트로 폴백
                # 일반 텍스트 처리
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    documents.append(Document(page_content=f.read(), metadata={"source": path}))
            else:
                # 일부 파일의 인코딩 문제가 전체 파이프라인을 중단하지 않도록 안전하게 로드
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
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


