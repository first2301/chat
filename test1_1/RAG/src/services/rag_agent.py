from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.document_loaders import WebBaseLoader
# from bs4 import SoupStrainer
from langchain_core.retrievers import BaseRetriever


class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 20,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "ko-llama-8B"
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model 
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()
        self.length_function = len
        self.is_separator_regex = False
        self.separators = ["\n\n", "\n", " ", ""]
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.retriever = self.retrieve(self.vector_store, self.k)
        self.llm = self.llm_model(self.ollama_model, self.retriever)
        self.prompt = self.prompt_template(self.prompt_template, self.question)
        self.rag_chain = self.rag_chain(self.llm, self.prompt)
        self.rag_chain.invoke({"question": self.question})
        self.search_kwargs = {"k": self.k}

        
    def web_loader(self, urls: list):
        loader = WebBaseLoader(
        web_paths=urls,
        # urls=urls,
        # browser="chrome",
        # headless=True,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
        },
        # bs_kwargs={
        #     "parse_only": SoupStrainer(["p", "h1", "h2", "h3", "div", "span"])  # 텍스트 노드만 파싱
        # }
        )
        return loader
    
    @staticmethod
    def text_splitter(self, docs: list):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,  # 텍스트 길이를 측정하는 함수
            is_separator_regex=self.is_separator_regex,  # 구분자가 정규식이 아님을 명시
            separators=self.separators
        )
        return splitter.split_documents(docs)

    def embedding(self, split_docs: list):
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in split_docs])
        return embeddings

    def save_vector_store(self, vector_store: FAISS, path: str):
        vector_store.save_local(path)
        return vector_store

    def load_vector_store(self, path: str):
        vector_store = self.vector_store_class.load_local(path)
        return vector_store

    def vector_store(self, split_docs: list):
        embeddings_data = self.embedding(split_docs)
        vector_store = self.vector_store_class.from_embeddings(embeddings_data, self.embedding_model)
        return vector_store


    def retrieve(self, vector_store: FAISS, k: int = 20):
        retriever = vector_store.as_retriever(search_kwargs=self.search_kwargs)
        return retriever
    
    @staticmethod
    def prompt_template(prompt_template: str, question: str):
        human_template = (
            """
            주어진 질문은 사실에 근거하여 답변하세요.\n
            답변은 한글로 작성하세요.\n
            답변은 주어진 질문에 대해 단계별로 생각해서 논리적으로 답변해 주세요.\n
            질문: {question}
            주어진 정보를 바탕으로 답변하세요.
            """
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt_template), # 시스템 프롬프트
            HumanMessagePromptTemplate.from_template(human_template) # 사용자 프롬프트
        ])
        return prompt

    @staticmethod
    def llm_model(llm_model: str, retriever: BaseRetriever, ollama_base_url: str, callback_manager: CallbackManager):
        llm = ChatOllama(
            model=llm_model,
            temperature=0.1,
            base_url=ollama_base_url,
            callback_manager=callback_manager,
            retriever=retriever
        )
        return llm

    @staticmethod
    def rag_chain(llm: ChatOllama, prompt: ChatPromptTemplate):
        chain = prompt | llm | StrOutputParser()
        return chain

    @staticmethod
    def rag_invoke(question: str, prompt_template: str, llm_model: str, retriever: BaseRetriever, ollama_base_url: str, callback_manager: CallbackManager, rag_chain: Runnable):
        prompt = prompt_template(prompt_template, question)
        llm = llm_model(llm_model, retriever, ollama_base_url, callback_manager)
        chain = rag_chain(llm, prompt) # RAG 체인
        return chain.invoke({"question": question})

