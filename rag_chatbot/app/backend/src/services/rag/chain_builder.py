"""체인 빌더: 프롬프트 생성 + 체인 구성 통합.

구성 요소:
- 기본 시스템/휴먼 메시지 템플릿을 생성합니다.
- 수동 체인(manual): `{context, question}` 입력을 받아 LLM으로 전달합니다.
- LCEL 체인: 리트리버 출력과 질문을 매핑하여 프롬프트→LLM→문자열 파서를 연결합니다.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class ChainBuilder:
    """프롬프트와 체인 구성 도우미.

    사용 방법:
        >>> cb = ChainBuilder()
        >>> manual = cb.build_manual_chain(llm)
        >>> lcel = cb.build_lcel_chain(retriever, llm)
    """

    def create_prompt(self) -> ChatPromptTemplate:
        """기본 RAG 프롬프트를 생성합니다.

        Returns:
            ChatPromptTemplate: 시스템/휴먼 메시지 템플릿으로 구성된 프롬프트
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
            HumanMessagePromptTemplate.from_template(human_template),
        ])

    def build_manual_chain(self, llm):
        """수동 컨텍스트 주입 체인.

        - 입력으로 `{context, question}` 맵을 받고, 프롬프트 → LLM → 문자열 파서로 이어집니다.

        Args:
            llm: LangChain 호환 LLM(예: `ChatOllama`)

        Returns:
            Runnable: 수동 체인
        """
        prompt = self.create_prompt()
        return prompt | llm | StrOutputParser()

    def build_lcel_chain(self, retriever, llm):
        """리트리버 결합 LCEL 체인.

        - 입력으로 질문 문자열을 받아, 내부에서 `{context: retriever, question: passthrough}`로 맵핑하여
          프롬프트 → LLM → 문자열 파서로 이어집니다.

        Args:
            retriever: LangChain 리트리버
            llm: LangChain 호환 LLM

        Returns:
            Runnable: LCEL 체인
        """
        prompt = self.create_prompt()
        return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


