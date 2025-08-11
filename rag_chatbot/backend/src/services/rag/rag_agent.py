"""RAG 에이전트 구현.

- 문서로부터 벡터 저장소를 구성하고, 리트리버-LLM 체인으로 질의를 처리합니다.
- 수동 컨텍스트 주입 체인과 LCEL 결합 체인을 모두 제공합니다.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.runnables import Runnable, RunnablePassthrough
from typing import List, Optional
import os
from rag_chatbot.backend.src.services.rag.config import Config
from rag_chatbot.backend.src.services.rag.ingestion_service import IngestionService
from rag_chatbot.backend.src.services.rag.vector_store_qdrant import VectorStoreManagerQdrant
from rag_chatbot.backend.src.services.rag.chain_builder import ChainBuilder

class RAGAgent:
    """RAG 에이전트 클래스.

    Args:
        document_paths: 초기 인덱싱용 문서 경로(파일/URL)
        embedding_model: 사전 생성된 임베딩 모델(미지정 시 Config에서 생성)
        chunk_size: 분할 크기(None이면 Config 값)
        chunk_overlap: 분할 중복 크기(None이면 Config 값)
        vector_store_class: 벡터 저장소 클래스(기본 FAISS)
        k: 검색 상위 k개(None이면 Config 값)
        ollama_base_url: Ollama 서버 URL(None이면 Config 값)
        ollama_model_name: Ollama 모델 이름(None이면 Config 값)
        vector_store_path: 벡터 저장소 경로(None이면 Config 값)

    Raises:
        ValueError: 벡터 저장소 경로와 문서가 모두 없는 경우(옵션 B 정책)
    """
    """
    RAG 에이전트: 문서로부터 벡터 저장소를 구성하고, 리트리버-LLM 체인으로 질의를 처리합니다.
    
    Args:
        document_paths: 문서 경로들
        embedding_model: 임베딩 모델
        chunk_size: 문서 분할 크기
        chunk_overlap: 문서 분할 중복 크기
        vector_store_class: 벡터 저장소 클래스
        k: 검색 결과 수
        ollama_base_url: Ollama 서버 URL
        ollama_model: Ollama 모델 이름
        vector_store_path: 벡터 저장소 경로
        
    Raises:
        ValueError: 벡터 저장소가 초기화되지 않았을 때 발생
    """
    def __init__(
        self,
        document_paths: Optional[List[str]] = None,
        embedding_model: Optional[HuggingFaceEmbeddings] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        vector_store_class=FAISS,
        k: Optional[int] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model_name: Optional[str] = None,
        vector_store_path: Optional[str] = None,
    ):
        # 기본 설정
        self.document_paths = document_paths or []
        self.chunk_size = chunk_size if chunk_size is not None else Config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else Config.chunk_overlap
        self.vector_store_class = vector_store_class
        self.k = k if k is not None else Config.k
        self.ollama_model = ollama_model_name or Config.ollama_model_name
        self.ollama_base_url = ollama_base_url or Config.ollama_base_url
        self.vector_store_path = vector_store_path or Config.vector_store_path
        
        # 콜백 매니저
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # 임베딩 모델 초기화
        self.embedding_model = embedding_model or Config.get_embedding_model()
        
        # 컴포넌트들 초기화
        self.vector_store = None
        self.retriever = None
        self.ingestion = IngestionService(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chain_builder = ChainBuilder()
        # LLM은 벡터스토어 유무와 무관하게 초기화하여 LLM-only 경로를 허용
        self.llm = ChatOllama(
            model=self.ollama_model,
            temperature=0.1,
            base_url=self.ollama_base_url,
            callback_manager=self.callback_manager,
        )
        self.chain = None
        
        # 벡터 저장소 로드 또는 생성
        self._initialize_vector_store()
        
        # RAG 체인 초기화
        self._initialize_rag_chain()
    
    def _initialize_vector_store(self):
        """
        벡터 저장소를 로드하거나 새로 생성합니다.
        
        Returns:
            None
        """
        # Qdrant 우선 사용: 컬렉션 존재/비어있음 정책에 따라 처리
        try:
            self.vector_store = VectorStoreManagerQdrant(embeddings=self.embedding_model).vs
            # 컬렉션이 없으면 생성 시도. langchain Qdrant 래퍼는 호출 시 자동 생성 지원.
            # document_paths 없고 정책이 auto_build이면 data_dir 스캔으로 생성 시도
            if not self.document_paths and Config.on_missing_vector_store == "auto_build":
                discovered = self.ingestion.discover_files()
                if discovered:
                    self.document_paths = discovered
                    self._create_vector_store_from_documents()
        except Exception:
            # Qdrant 연결 실패 시 기존 FAISS 경로(하위호환)로 폴백
            if not self.vector_store_path and not self.document_paths:
                # LLM-only 허용
                return
            if self.vector_store_path and os.path.exists(self.vector_store_path):
                try:
                    self.vector_store = self.vector_store_class.load_local(
                        self.vector_store_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                except TypeError:
                    self.vector_store = self.vector_store_class.load_local(
                        self.vector_store_path,
                        self.embedding_model,
                    )
            else:
                if self.document_paths:
                    self._create_vector_store_from_documents()
    
    def _create_vector_store_from_documents(self):
        """
        문서들로부터 벡터 저장소를 생성합니다.
        
        Returns:
            None
        """
        # 문서 로드
        documents = []
        file_paths = [p for p in self.document_paths if not p.startswith(("http://", "https://"))]
        url_paths = [p for p in self.document_paths if p.startswith(("http://", "https://"))]
        if file_paths:
            documents.extend(self.ingestion.load_files(file_paths))
        if url_paths:
            documents.extend(self.ingestion.load_urls(url_paths))
        
        # 텍스트 분할
        split_docs = self.ingestion.split(documents)
        
        # Qdrant가 초기화되어 있으면 업서트, 아니면 로컬 FAISS 생성
        if self.vector_store is not None and hasattr(self.vector_store, "add_documents"):
            self.vector_store.add_documents(split_docs)
        else:
            self.vector_store = self.vector_store_class.from_documents(
                split_docs, 
                self.embedding_model
            )
            # Qdrant는 원격 저장이므로 별도 저장 불필요. FAISS 폴백일 때만 저장.
            if self.vector_store_path and hasattr(self.vector_store, "save_local"):
                self.vector_store.save_local(self.vector_store_path)
    
    # 로더/스플리터/발견은 전담 컴포넌트가 담당
    
    def _initialize_rag_chain(self):
        """
        RAG 체인을 초기화합니다.
        
        - 벡터스토어가 없는 경우(LLM-only 모드) 체인 생성을 건너뜁니다.
        """
        if self.vector_store is None:
            # 벡터스토어 없으면 검색 체인을 구성하지 않음(LLM-only 모드)
            self.retriever = None
            self.chain = None
            self.lcel_chain = None
            return
        
        # 검색기 생성
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        
        # 프롬프트 템플릿 생성
        manual_chain = self.chain_builder.build_manual_chain(self.llm)
        
        # 수동 컨텍스트 주입 체인
        self.chain = manual_chain

        # LCEL 리트리버 결합 체인
        self.lcel_chain = self.chain_builder.build_lcel_chain(self.retriever | self._format_docs, self.llm)
    
    # 프롬프트는 팩토리가 담당

    @staticmethod
    def _format_docs(docs: List) -> str:
        """
        문서들을 포맷팅합니다.
        
        Returns:
            str: 포맷팅된 문서
        """
        return "\n\n".join([doc.page_content for doc in docs])
    
    def add_documents(self, document_paths: List[str]):
        """
        새로운 문서들을 추가합니다.
        
        Returns:
            None
        """
        self.document_paths.extend(document_paths)
        self._create_vector_store_from_documents()
        self._initialize_rag_chain()
    
    def query(self, question: str) -> str:
        """
        질문에 답변합니다.
        
        Returns:
            str: 답변
        """
        if self.chain is None:
            raise ValueError("RAG 체인이 초기화되지 않았습니다.")
        
        # 관련 문서 검색
        relevant_docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 답변 생성
        response = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        return response

    def query_lcel(self, question: str) -> str:
        """리트리버를 LCEL 체인에 결합해 질의를 처리합니다."""
        if self.lcel_chain is None:
            raise ValueError("LCEL 체인이 초기화되지 않았습니다.")
        return self.lcel_chain.invoke(question)
    
    def save_vector_store(self):
        """벡터 저장소를 저장합니다 (.env의 VECTOR_STORE_PATH 필수)."""
        # Qdrant 모드에서는 별도 저장이 필요 없습니다. FAISS 폴백만 수행.
        if hasattr(self.vector_store, "save_local"):
            if not self.vector_store_path:
                raise ValueError("VECTOR_STORE_PATH가 설정되지 않았습니다. .env에 경로를 지정하세요.")
            if self.vector_store is None:
                raise ValueError("저장할 벡터 저장소가 없습니다.")
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
    
    def load_vector_store(self):
        """벡터 저장소를 로드합니다 (.env의 VECTOR_STORE_PATH 필수)."""
        # Qdrant 모드에서는 컬렉션 연결로 충분합니다. 폴백인 경우에만 디스크에서 로딩.
        try:
            self.vector_store = VectorStoreManagerQdrant(embeddings=self.embedding_model).vs
        except Exception:
            if not self.vector_store_path:
                raise ValueError("VECTOR_STORE_PATH가 설정되지 않았습니다. .env에 경로를 지정하세요.")
            try:
                self.vector_store = self.vector_store_class.load_local(
                    self.vector_store_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                self.vector_store = self.vector_store_class.load_local(
                    self.vector_store_path,
                    self.embedding_model,
                )
        self._initialize_rag_chain()

    # 증분 인덱싱을 위한 간단한 퍼블릭 API
    def add_files(self, paths: List[str]):
        self.add_documents(paths)

    def add_urls(self, urls: List[str]):
        self.add_documents(urls)

    def llm_query(self, question: str) -> str:
        """LLM을 직접 호출하여 답변 텍스트를 반환합니다."""
        result = self.llm.invoke(question)
        return getattr(result, "content", str(result))

# 사용 예시를 위한 헬퍼 함수들
def create_rag_agent_from_documents(
    document_paths: List[str],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    **kwargs,
) -> RAGAgent:
    """문서들로부터 RAG 에이전트를 생성합니다."""
    return RAGAgent(document_paths=document_paths, embedding_model=embedding_model, **kwargs)


def create_rag_agent_from_env_vector_store(
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    **kwargs,
) -> RAGAgent:
    """.env의 VECTOR_STORE_PATH를 사용하여 RAG 에이전트를 생성합니다."""
    return RAGAgent(embedding_model=embedding_model, **kwargs)