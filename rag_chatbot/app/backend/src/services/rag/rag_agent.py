"""RAG 에이전트 구현.

- 문서로부터 벡터 저장소(Qdrant)를 구성하고, 리트리버-LLM 체인으로 질의를 처리합니다.
- 수동 컨텍스트 주입 체인과 LCEL 결합 체인을 모두 제공합니다.
- LLM-only 경로를 지원하여 벡터스토어 미가용 시에도 기본 응답이 가능합니다.
"""

from langchain_ollama import ChatOllama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from typing import List
from backend.src.services.rag.config import Config
from backend.src.services.rag.ingestion_service import IngestionService
from backend.src.services.rag.vector_store_qdrant import VectorStoreManagerQdrant
from backend.src.services.rag.chain_builder import ChainBuilder

class RAGAgent:
    """RAG 에이전트 클래스.

    Attributes:
        document_paths: 인덱싱 대상 문서/URL 경로 목록
        chunk_size: 문서 분할 크기(문자 기준)
        chunk_overlap: 분할 중복 크기
        k: 검색 상위 k개
        ollama_temperature: LLM 온도
        ollama_model: Ollama 모델명
        ollama_base_url: Ollama 서버 URL
        callback_manager: LangChain 콜백 매니저(스트리밍 로그 등)
        embedding_model: 임베딩 모델(HuggingFaceEmbeddings)
        vector_store: Qdrant 기반 벡터 스토어 인스턴스
        retriever: LangChain 리트리버
        ingestion: 인제스천 서비스(로딩/분할)
        chain_builder: 체인 빌더(프롬프트/체인 구성)
        llm: ChatOllama 인스턴스
        chain: 수동 컨텍스트 주입 체인
        lcel_chain: 리트리버 결합 LCEL 체인

    Raises:
        RuntimeError: Qdrant 벡터 스토어 초기화 실패 시
    """
    
    def __init__(
        self
    ):
        # 기본 설정
        self.document_paths = []
        self.chunk_size = Config.chunk_size
        self.chunk_overlap = Config.chunk_overlap
        self.k = Config.k
        self.ollama_temperature = Config.ollama_temperature
        self.ollama_model = Config.ollama_model_name
        self.ollama_base_url = Config.ollama_base_url
        
        # 콜백 매니저
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # 임베딩 모델 초기화
        self.embedding_model = Config.get_embedding_model()
        
        # 컴포넌트들 초기화
        self.vector_store = None
        self.retriever = None
        self.ingestion = IngestionService(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chain_builder = ChainBuilder()
        # LLM은 벡터스토어 유무와 무관하게 초기화하여 LLM-only 경로를 허용
        self.llm = ChatOllama(
            model=self.ollama_model,
            temperature=self.ollama_temperature,
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

        동작 개요:
        - Qdrant 전용 정책: Qdrant 연결 성공 시 해당 컬렉션을 사용합니다.
          - `Config.on_missing_vector_store == "auto_build"`이고 초기 문서 경로가 비어 있으면
            `IngestionService.discover_files()`로 DATA_DIR을 스캔해 자동 인덱싱을 시도합니다.
        - Qdrant 연결 실패 시 적절한 에러 메시지 제공

        Returns:
            None
        """
        # Qdrant 전용 사용: 컬렉션 존재/비어있음 정책에 따라 처리
        try:
            self.vector_store = VectorStoreManagerQdrant(embeddings=self.embedding_model).vs
            # 컬렉션이 없으면 생성 시도. langchain Qdrant 래퍼는 호출 시 자동 생성 지원.
            # document_paths 없고 정책이 auto_build이면 data_dir 스캔으로 생성 시도
            if not self.document_paths and Config.on_missing_vector_store == "auto_build":
                discovered = self.ingestion.discover_files()
                if discovered:
                    self.document_paths = discovered
                    self._create_vector_store_from_documents()
        except Exception as e:
            # Qdrant 전용 모드이므로 폴백 없이 에러 전파
            raise RuntimeError(f"Qdrant 벡터 스토어 초기화 실패: {e}")
    
    def _create_vector_store_from_documents(self):
        """
        `self.document_paths`에 지정된 데이터 소스로부터 문서를 로드·분할한 뒤
        임베딩하여 벡터 저장소에 업서트합니다.

        동작 개요:
        - 입력 소스 분류: 로컬 파일과 URL을 분리 처리합니다.
          - 파일: PDF(.pdf)은 PyMuPDF 기반 로더, 그 외 텍스트 파일은 UTF-8로 읽어 처리합니다.
          - URL: LangChain `WebBaseLoader`로 HTTP 수집합니다(robots/속도 제한 고려 필요).
        - 문서 분할: `IngestionService`의 `RecursiveCharacterTextSplitter`를 사용하여
          `Config.chunk_size`와 `Config.chunk_overlap`에 따라 분할합니다.
        - 저장소 쓰기: Qdrant 벡터 스토어에 `add_documents`로 업서트합니다.

        주의/제약:
        - 중복 제어를 별도로 수행하지 않습니다. 동일 문서를 반복 호출 시 중복이 발생할 수 있습니다.
        - PDF 처리를 위해서는 `pymupdf`(PyMuPDF) 의존성이 필요합니다.
        - 네트워크 수집은 외부 환경에 의존하므로 실패 시 예외가 전파될 수 있습니다.

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

        # Qdrant 벡터 스토어에 배치로 문서 추가
        if self.vector_store is not None and hasattr(self.vector_store, "add_documents"):
            batch_size = getattr(Config, "upsert_batch_size", 200)
            if batch_size and batch_size > 0:
                for i in range(0, len(split_docs), batch_size):
                    batch = split_docs[i:i + batch_size]
                    if batch:
                        self.vector_store.add_documents(batch)
            else:
                self.vector_store.add_documents(split_docs)
        else:
            raise RuntimeError("Qdrant 벡터 스토어가 초기화되지 않았습니다.")
    
    # 로더/스플리터/발견은 전담 컴포넌트가 담당
    
    def _initialize_rag_chain(self):
        """
        RAG 체인을 초기화합니다.

        동작 개요:
        - 벡터스토어가 없으면 검색 체인을 구성하지 않고 LLM-only 모드로 설정합니다.
        - 벡터스토어가 있으면 `as_retriever(k=self.k)`로 검색기를 만들고,
          프롬프트/체인은 `ChainBuilder`를 통해 생성합니다.
          - 수동 체인(manual): 단순 컨텍스트 주입 방식으로 `self.chain`에 설정됩니다.
          - LCEL 체인: 리트리버 출력 → 문서 포맷팅 → LLM 호출을 연결하여 `self.lcel_chain`에 설정됩니다.

        Returns:
            None
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
        새로운 문서 경로를 추가하고 벡터 저장소에 반영한 뒤 체인을 재초기화합니다.

        주의/제약:
        - 중복 방지는 내부적으로 수행하지 않습니다. 동일 경로를 여러 번 추가하면 중복 인덱싱이 발생할 수 있습니다.
        - 원격(Qdrant) 사용 시 `add_documents`로 업서트가 수행됩니다.

        Args:
            document_paths: 파일 경로 또는 URL 목록

        Returns:
            None
        """
        self.document_paths.extend(document_paths)
        self._create_vector_store_from_documents()
        self._initialize_rag_chain()
    
    def query(self, question: str) -> str:
        """
        수동 컨텍스트 주입 체인으로 질의를 처리하여 답변 문자열을 반환합니다.

        동작 개요:
        - 리트리버로 관련 문서 `k`개를 조회하고, 본문을 합쳐 컨텍스트를 구성합니다.
        - 프롬프트에 `{context, question}`을 주입하여 LLM 응답을 생성합니다.

        Raises:
            ValueError: 체인이 초기화되지 않았을 때(벡터스토어 없음 등)

        Returns:
            str: LLM 응답 텍스트
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
        """
        LCEL 체인(리트리버 → 문서 포맷팅 → LLM)으로 질의를 처리합니다.

        Raises:
            ValueError: LCEL 체인이 초기화되지 않은 경우

        Returns:
            str: LLM 응답 텍스트
        """
        if self.lcel_chain is None:
            raise ValueError("LCEL 체인이 초기화되지 않았습니다.")
        return self.lcel_chain.invoke(question)
    
    def save_vector_store(self):
        """
        벡터 저장소를 저장합니다.

        참고:
        - Qdrant(원격) 사용 시 별도 저장이 필요 없습니다.
        - 이 메서드는 Qdrant 전용 모드에서는 동작하지 않습니다.

        Raises:
            RuntimeError: Qdrant 전용 모드에서는 사용할 수 없습니다.
        """
        # Qdrant 전용 모드에서는 별도 저장이 필요 없습니다.
        raise RuntimeError("Qdrant 전용 모드에서는 save_vector_store를 사용할 수 없습니다. Qdrant는 자동으로 데이터를 저장합니다.")
    
    def load_vector_store(self):
        """
        벡터 저장소를 다시 로드합니다.

        동작 개요:
        - Qdrant 전용: 원격 컬렉션 연결을 시도합니다.
        - Qdrant 연결 실패 시 적절한 에러 메시지 제공

        Raises:
            RuntimeError: Qdrant 연결 실패 시

        Returns:
            None
        """
        try:
            # Qdrant 연결 시도
            self.vector_store = VectorStoreManagerQdrant(embeddings=self.embedding_model).vs
        except Exception as e:
            # Qdrant 연결 실패 시 적절한 에러 처리
            raise RuntimeError(f"Qdrant 연결 실패: {e}. Qdrant 서비스 상태를 확인하세요.")
        
        self._initialize_rag_chain()

    # 증분 인덱싱을 위한 간단한 퍼블릭 API
    def add_files(self, paths: List[str]):
        """파일 경로 목록을 추가 인덱싱합니다. URL은 허용되지 않습니다.

        Args:
            paths: 파일 시스템 경로 목록
        """
        self.add_documents(paths)

    def add_urls(self, urls: List[str]):
        """URL 목록을 추가 인덱싱합니다.

        Args:
            urls: 웹 문서 URL 목록
        """
        self.add_documents(urls)

    def llm_query(self, question: str) -> str:
        """
        검색 없이 LLM만 직접 호출하여 응답 텍스트를 반환합니다.

        주의:
        - 리트리버 컨텍스트가 없으므로 RAG 품질과 무관한 순수 LLM 응답입니다.

        Returns:
            str: LLM 응답 텍스트
        """
        result = self.llm.invoke(question)
        return getattr(result, "content", str(result))

# # 사용 예시를 위한 헬퍼 함수들
# def create_rag_agent_from_documents(
#     document_paths: List[str],
#     embedding_model: Optional[HuggingFaceEmbeddings] = None,
#     **kwargs,
# ) -> RAGAgent:
#     """문서들로부터 RAG 에이전트를 생성합니다."""
#     return RAGAgent(document_paths=document_paths, embedding_model=embedding_model, **kwargs)


# def create_rag_agent_from_env_vector_store(
#     embedding_model: Optional[HuggingFaceEmbeddings] = None,
#     **kwargs,
# ) -> RAGAgent:
#     """.env의 VECTOR_STORE_PATH를 사용하여 RAG 에이전트를 생성합니다."""
#     return RAGAgent(embedding_model=embedding_model, **kwargs)