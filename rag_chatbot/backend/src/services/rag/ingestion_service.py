"""인제스천 서비스: 파일 스캔 + 로딩 + 분할 통합."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_chatbot.backend.src.services.rag.config import Config


class IngestionService:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""],
        )
        # rag_chatbot 루트 디렉토리 계산
        self._rag_root = Path(__file__).resolve().parents[5]

    def discover_files(self, data_dir: str | None = None, patterns: List[str] | None = None) -> List[str]:
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
        for path in paths:
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


