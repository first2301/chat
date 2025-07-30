# rag_ollama_chroma.py - 수정된 버전
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
from pathlib import Path

# 경로 설정
root_path = Path(os.getcwd()).parent.parent
embeddings_path = root_path / "ollama-service" / "models" / "BGE-m3-ko"
print(f"임베딩 모델 경로: {embeddings_path}")

# HuggingFace 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(
    model_name=str(embeddings_path)
)

# 벡터 스토어 로드
vectorstore = FAISS.load_local("vector_db/pcn_web", embedding_model, allow_dangerous_deserialization=True)

# 검색기(Retriever) 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# 올바른 ChatOllama 설정
llm = ChatOllama(
    model="ko_llama",  # Ollama 모델 이름 (파일 경로가 아님)
    temperature=0.1,
    base_url="http://localhost:11435",  # Host 모드에서는 localhost 사용
    callback_manager=callback_manager,
)

# 프롬프트 설정
system_template = (
    "당신은 친절하고 유능한 AI 어시스턴트입니다. "
    "사용자의 질문에 대해 신뢰할 수 있는 정보를 바탕으로 정확하고 간결하게 답변하세요."
)
human_template = (
    "아래의 질문에 대해 단계별로 논리적으로 생각하여 답변해 주세요.\n"
    "질문: {question}"
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = prompt | llm

# RAG 체인 구성
template = """다음 맥락을 사용하여 질문에 답하세요:
{context}

질문: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("RAG 체인 구성 완료. 이제 질문할 수 있습니다!")

# 질의 응답 테스트
def ask_question(question):
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"오류 발생: {e}. Ollama 모델이 제대로 실행 중인지 확인하세요."

# 테스트
if __name__ == "__main__":
    question = "피씨엔 소개"
    print(f"질문: {question}")
    response = ask_question(question)
    print(f"답변: {response}") 