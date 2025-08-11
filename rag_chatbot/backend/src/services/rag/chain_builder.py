"""체인 빌더: 프롬프트 생성 + 체인 구성 통합."""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class ChainBuilder:
    def create_prompt(self) -> ChatPromptTemplate:
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
        prompt = self.create_prompt()
        return prompt | llm | StrOutputParser()

    def build_lcel_chain(self, retriever, llm):
        prompt = self.create_prompt()
        return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


