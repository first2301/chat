from langsmith.evaluation import EvaluationResult, RunEvaluator, RunEvaluatorChain, evaluate_runs
from langsmith.schemas import Example, Run, Feedback
from typing import List, Dict, Any

class RAGEvaluator:
    """
    LangSmith 기반 RAG 평가 클래스
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

    def create_example(self, input: str, output: str, retrieval_context: List[str], expected_output: str) -> Example:
        """
        LangSmith Example 객체 생성
        """
        example = Example(
            inputs={"question": input, "retrieval_context": retrieval_context},
            outputs={"answer": output},
            expected_output=expected_output
        )
        return example

    def create_run(self, example: Example, run_id: str = None) -> Run:
        """
        LangSmith Run 객체 생성
        """
        run = Run(
            id=run_id or "run_" + example.inputs["question"][:10],
            inputs=example.inputs,
            outputs=example.outputs,
            reference_output=example.expected_output
        )
        return run

    def evaluate_run(self, run: Run, evaluators: List[RunEvaluator]) -> List[EvaluationResult]:
        """
        Run에 대해 평가 수행
        """
        results = []
        for evaluator in evaluators:
            result = evaluator.evaluate_run(run)
            results.append(result)
        return results

    def get_default_evaluators(self) -> List[RunEvaluator]:
        """
        LangSmith에서 제공하는 기본 RAG 평가자 체인 반환
        """
        # 실제로는 LangSmith에서 제공하는 평가자 체인을 사용해야 함
        # 예시로 커스텀 evaluator를 생성
        class DummyRelevancyEvaluator(RunEvaluator):
            def evaluate_run(self, run: Run) -> EvaluationResult:
                # 실제 평가 로직은 LangSmith의 평가자 사용
                return EvaluationResult(
                    key="relevancy",
                    score=1.0,
                    comment="관련성 높음"
                )
        class DummyPrecisionEvaluator(RunEvaluator):
            def evaluate_run(self, run: Run) -> EvaluationResult:
                return EvaluationResult(
                    key="precision",
                    score=1.0,
                    comment="정밀도 높음"
                )
        class DummyRecallEvaluator(RunEvaluator):
            def evaluate_run(self, run: Run) -> EvaluationResult:
                return EvaluationResult(
                    key="recall",
                    score=1.0,
                    comment="재현율 높음"
                )
        return [DummyRelevancyEvaluator(), DummyPrecisionEvaluator(), DummyRecallEvaluator()]

    def evaluate(self, input: str, output: str, retrieval_context: List[str], expected_output: str) -> List[EvaluationResult]:
        """
        전체 평가 파이프라인 실행
        """
        example = self.create_example(input, output, retrieval_context, expected_output)
        run = self.create_run(example)
        evaluators = self.get_default_evaluators()
        results = self.evaluate_run(run, evaluators)
        return results

    def evaluate_batch(self, examples: List[Dict[str, Any]]) -> List[List[EvaluationResult]]:
        """
        여러 예제에 대해 배치 평가 실행
        """
        results = []
        for ex in examples:
            res = self.evaluate(
                ex["input"],
                ex["output"],
                ex.get("retrieval_context", []),
                ex.get("expected_output", "")
            )
            results.append(res)
        return results