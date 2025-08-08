"""RAG 에이전트 구현.

- 문서로부터 벡터 저장소를 구성하고, 리트리버-LLM 체인으로 질의를 처리합니다.
- 수동 컨텍스트 주입 체인과 LCEL 결합 체인을 모두 제공합니다.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from typing import List, Optional
import os
from rag_chatbot.backend.src.services.rag.config import Config

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
        
        # 텍스트 분할 설정
        self.length_function = len
        self.is_separator_regex = False
        self.separators = ["\n\n", "\n", " ", ""]
        
        # 콜백 매니저
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # 임베딩 모델 초기화
        self.embedding_model = embedding_model or Config.get_embedding_model()
        
        # 컴포넌트들 초기화
        self.vector_store = None
        self.retriever = None
        self.llm = None
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
        # .env의 VECTOR_STORE_PATH만 사용. 경로가 없고 문서도 없으면 에러.
        if not self.vector_store_path and not self.document_paths:
            raise ValueError("VECTOR_STORE_PATH가 .env에 설정되어 있지 않고 document_paths도 제공되지 않았습니다.")

        if self.vector_store_path and os.path.exists(self.vector_store_path):
            # 기존 벡터 저장소 로드 (버전 호환 옵션 포함)
            try:
                self.vector_store = self.vector_store_class.load_local(
                    self.vector_store_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                # allow_dangerous_deserialization 미지원 버전 호환
                self.vector_store = self.vector_store_class.load_local(
                    self.vector_store_path,
                    self.embedding_model,
                )
        else:
            # 새로 생성 (문서가 있는 경우)
            if self.document_paths:
                self._create_vector_store_from_documents()
    
    def _create_vector_store_from_documents(self):
        """
        문서들로부터 벡터 저장소를 생성합니다.
        
        Returns:
            None
        """
        # 문서 로드
        documents = self._load_documents()
        
        # 텍스트 분할
        split_docs = self._split_documents(documents)
        
        # 벡터 저장소 생성
        self.vector_store = self.vector_store_class.from_documents(
            split_docs, 
            self.embedding_model
        )
        
        # 저장 (경로가 지정된 경우)
        if self.vector_store_path:
            self.vector_store.save_local(self.vector_store_path)
    
    def _load_documents(self) -> List:
        """
        문서들을 로드합니다.
        
        Returns:
            List: 문서 리스트
        """
        documents = []
        
        for path in self.document_paths:
            if path.startswith(("http://", "https://")):
                # 웹 문서 로드
                loader = self._create_web_loader([path])
                documents.extend(loader.load())
            else:
                # 로컬 파일 로드: PDF 우선 처리, 그 외 텍스트 파일로 가정
                if path.lower().endswith(".pdf"):
                    loader = PyMuPDFLoader(path)
                    documents.extend(loader.load())
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        from langchain_core.documents import Document
                        documents.append(Document(page_content=f.read(), metadata={"source": path}))
        
        return documents
    
    def _create_web_loader(self, urls: List[str]):
        """
        웹 로더를 생성합니다.
        
        Returns:
            WebBaseLoader: 웹 로더
        """
        return WebBaseLoader(
            web_paths=urls,
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
            }
        )
    
    def _split_documents(self, documents: List):
        """
        문서들을 분할합니다.
        
        Returns:
            List: 분할된 문서 리스트
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=self.is_separator_regex,
            separators=self.separators
        )
        return splitter.split_documents(documents)
    
    def _initialize_rag_chain(self):
        """
        RAG 체인을 초기화합니다.
        
        Returns:
            None
        """
        if self.vector_store is None:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다. 문서를 먼저 로드해주세요.")
        
        # 검색기 생성
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        
        # LLM 생성
        self.llm = ChatOllama(
            model=self.ollama_model,
            temperature=0.1,
            base_url=self.ollama_base_url,
            callback_manager=self.callback_manager
        )
        
        # 프롬프트 템플릿 생성
        prompt = self._create_prompt_template()
        
        # 수동 컨텍스트 주입 체인
        self.chain = prompt | self.llm | StrOutputParser()

        # LCEL 리트리버 결합 체인
        self.lcel_chain = {
            "context": self.retriever | self._format_docs,
            "question": RunnablePassthrough(),
        } | prompt | self.llm | StrOutputParser()
    
    def _create_prompt_template(self):
        """
        프롬프트 템플릿을 생성합니다.
        
        Returns:
            ChatPromptTemplate: 프롬프트 템플릿
        """
        system_template = (
            "당신은 도움이 되는 AI 어시스턴트입니다. "
            "주어진 컨텍스트를 바탕으로 질문에 답변해주세요. "
            "답변은 한글로 작성하고, 사실에 근거하여 논리적으로 답변해주세요."
        )
        
        human_template = (
            "컨텍스트: {context}\n\n"
            "질문: {question}\n\n"
            "위의 컨텍스트를 바탕으로 질문에 답변해주세요."
        )
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

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
        if not self.vector_store_path:
            raise ValueError("VECTOR_STORE_PATH가 설정되지 않았습니다. .env에 경로를 지정하세요.")
        if self.vector_store is None:
            raise ValueError("저장할 벡터 저장소가 없습니다.")
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
    
    def load_vector_store(self):
        """벡터 저장소를 로드합니다 (.env의 VECTOR_STORE_PATH 필수)."""
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