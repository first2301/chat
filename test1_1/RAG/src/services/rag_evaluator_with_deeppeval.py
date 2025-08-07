from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate, AsyncConfig
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)


class RAGEvaluator:
    """
    RAG 평가 클래스
    Args:
        ollama_model: Ollama 모델 이름
        ollama_base_url: Ollama 베이스 URL
        temperature: 온도(temperature)
    """
    def __init__(
        self,
        ollama_model: str,
        ollama_base_url: str,
        temperature: float
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature

    def create_ollama_model(self, ollama_model: str, ollama_base_url: str, temperature: float):
        """
        Ollama 모델 초기화
        Args:
            ollama_model: Ollama 모델 이름
            ollama_base_url: Ollama 베이스 URL
            temperature: 온도(temperature)
        Returns:
            ollama_llm: Ollama 모델
        """
        ollama_llm = OllamaModel(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature
        )
        return ollama_llm
    
    def answer_relevancy(self, ollama_llm: OllamaModel):
        """
        답변 관련성 평가
        Args:
            ollama_llm: Ollama 모델
        Returns:
            answer_correctness: 답변 관련성 평가
        """
        answer_correctness = AnswerRelevancyMetric(model=ollama_llm)
        return answer_correctness
    
    def citation_accuracy(self, ollama_llm: OllamaModel):
        """
        인용구 정확도 평가
        Args:
            ollama_llm: Ollama 모델
        Returns:
            citation_accuracy: 인용구 정확도 평가
        """
        citation_accuracy = CitationAccuracyMetric(model=ollama_llm)
        return citation_accuracy
    
    def contextual_relevancy(self, ollama_llm: OllamaModel, retrieval_context: list):
        """
        컨텍스트 관련성 평가
        Args:
            ollama_llm: Ollama 모델
            retrieval_context: 검색된 컨텍스트
        """
        contextual_relevancy = ContextualRelevancyMetric(model=ollama_llm)
        return contextual_relevancy
    
    def contextual_recall(self, ollama_llm: OllamaModel, retrieval_context: list):
        """
        컨텍스트 재현성 평가
        Args:
            ollama_llm: Ollama 모델
            retrieval_context: 검색된 컨텍스트
        """
        contextual_recall = ContextualRecallMetric(model=ollama_llm)
        return contextual_recall
    
    def contextual_precision(self, ollama_llm: OllamaModel, retrieval_context: list):
        """
        컨텍스트 정밀도 평가
        Args:
            ollama_llm: Ollama 모델
            retrieval_context: 검색된 컨텍스트
        """
        contextual_precision = ContextualPrecisionMetric(model=ollama_llm)
        return contextual_precision
    
    def evaluate_answer(self, ollama_llm: OllamaModel, retrieval_context: list):
        """
        답변 평가
        """
        answer_correctness = self.answer_relevancy(ollama_llm)
        citation_accuracy = self.citation_accuracy(ollama_llm)
        return answer_correctness, citation_accuracy
    
    def evaluate_context(self, ollama_llm: OllamaModel, retrieval_context: list):
        """
        컨텍스트 평가
        """
        relevancy = self.contextual_relevancy(ollama_llm, retrieval_context)
        recall = self.contextual_recall(ollama_llm, retrieval_context)
        precision = self.contextual_precision(ollama_llm, retrieval_context)
        return relevancy, recall, precision
    
    def evaluate_all(self, ollama_llm: OllamaModel, retrieval_context: list):
        """
        모든 평가
        """
        answer_correctness, citation_accuracy = self.evaluate_answer(ollama_llm, retrieval_context)
        relevancy, recall, precision = self.evaluate_context(ollama_llm, retrieval_context)
        return answer_correctness, citation_accuracy, relevancy, recall, precision
    
    def create_test_case(self, input: str, actual_output: str, retrieval_context: list, expected_output: str):
        """
        테스트 케이스 생성
        """
        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            expected_output=expected_output
        )
        return test_case
    
    def evaluate_test_case(self, test_case: LLMTestCase, answer_correctness: AnswerRelevancyMetric, citation_accuracy: CitationAccuracyMetric, relevancy: ContextualRelevancyMetric, recall: ContextualRecallMetric, precision: ContextualPrecisionMetric):
        """
        테스트 케이스 평가
        """
        result = evaluate(
            [test_case],
            [answer_correctness, citation_accuracy],
            async_config=AsyncConfig(run_async=False))
        return result