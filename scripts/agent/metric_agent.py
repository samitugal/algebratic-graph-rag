from typing_extensions import override
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from scripts.agent.base_agent import BaseAgent
from scripts.model.language_models.openai_client import OpenAIClient


class MetricScore(BaseModel):
    metric_name: str = Field(description="Name of the metric")
    score: float = Field(description="Score between 0.0 and 1.0")
    explanation: str = Field(description="Detailed explanation of the score")
    evidence: str = Field(description="Specific evidence supporting the score")


class MetricEvaluation(BaseModel):
    overall_score: float = Field(description="Overall weighted score")
    individual_scores: List[MetricScore] = Field(description="Individual metric scores")
    summary: str = Field(description="Overall evaluation summary")


class MetricAgent(BaseAgent):
    def __init__(self, client: OpenAIClient):
        super().__init__(client)

    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> str:
        """Evaluate if the answer is faithful to the retrieved context"""
        prompt = f"""
        # Faithfulness Evaluation

        ## Task
        Evaluate how faithful the answer is to the provided context. The answer should only contain information that can be directly inferred or stated from the context.

        ## Scoring Guidelines
        - 1.0: Answer is completely faithful, all information comes from context
        - 0.8: Answer is mostly faithful with minor extrapolations
        - 0.6: Answer contains some information not in context but mostly accurate
        - 0.4: Answer has significant unfaithful information
        - 0.2: Answer is largely unfaithful to context
        - 0.0: Answer completely contradicts or ignores context

        ## Question
        {question}

        ## Context
        {context}

        ## Answer
        {answer}

        Provide a score between 0.0 and 1.0, explanation, and specific evidence.
        """

        return self.client.generate_text(prompt, response_json=False)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> str:
        """Evaluate how relevant the answer is to the question"""
        prompt = f"""
        # Answer Relevancy Evaluation

        ## Task
        Evaluate how relevant and directly responsive the answer is to the question asked.

        ## Scoring Guidelines
        - 1.0: Answer directly and completely addresses the question
        - 0.8: Answer addresses most aspects of the question
        - 0.6: Answer partially addresses the question
        - 0.4: Answer somewhat relates to the question
        - 0.2: Answer barely relates to the question
        - 0.0: Answer is completely irrelevant

        ## Question
        {question}

        ## Answer
        {answer}

        Provide a score between 0.0 and 1.0, explanation, and specific evidence.
        """

        return self.client.generate_text(prompt, response_json=False)

    def evaluate_context_precision(self, question: str, context: str) -> str:
        """Evaluate if the retrieved context is relevant to the question"""
        prompt = f"""
        # Context Precision Evaluation

        ## Task
        Evaluate how precise and relevant the retrieved context is for answering the question.

        ## Scoring Guidelines
        - 1.0: All context is highly relevant and useful
        - 0.8: Most context is relevant with minimal noise
        - 0.6: Context is mostly relevant but contains some irrelevant information
        - 0.4: Context contains equal amounts of relevant and irrelevant information
        - 0.2: Context is mostly irrelevant
        - 0.0: Context is completely irrelevant

        ## Question
        {question}

        ## Context
        {context}

        Provide a score between 0.0 and 1.0, explanation, and specific evidence.
        """

        return self.client.generate_text(prompt, response_json=False)

    def evaluate_completeness(self, question: str, answer: str, context: str) -> str:
        """Evaluate how complete the answer is"""
        prompt = f"""
        # Completeness Evaluation

        ## Task
        Evaluate how complete and comprehensive the answer is in addressing all aspects of the question.

        ## Scoring Guidelines
        - 1.0: Answer is comprehensive and addresses all aspects
        - 0.8: Answer addresses most important aspects
        - 0.6: Answer covers main points but misses some details
        - 0.4: Answer is partial and misses important information
        - 0.2: Answer is incomplete and superficial
        - 0.0: Answer fails to address the question

        ## Question
        {question}

        ## Available Context
        {context}

        ## Answer
        {answer}

        Provide a score between 0.0 and 1.0, explanation, and specific evidence.
        """

        return self.client.generate_text(prompt, response_json=False)

    def evaluate_hallucination(self, question: str, answer: str, context: str) -> str:
        """Detect hallucinations in the answer"""
        prompt = f"""
        # Hallucination Detection

        ## Task
        Detect if the answer contains hallucinated (made-up) information that cannot be verified from the context.

        ## Scoring Guidelines (Lower is better for hallucination)
        - 0.0: No hallucinations detected, all information verifiable
        - 0.2: Minor unverifiable details
        - 0.4: Some hallucinated information present
        - 0.6: Significant hallucinations
        - 0.8: Major hallucinations throughout
        - 1.0: Answer is mostly hallucinated

        ## Question
        {question}

        ## Context
        {context}

        ## Answer
        {answer}

        Provide a score between 0.0 and 1.0, explanation, and specific evidence of any hallucinations.
        """

        return self.client.generate_text(prompt, response_json=False)

    @override
    def invoke(
        self,
        question: str,
        answer: str,
        context: str,
        response_json: bool = True,
        response_model: type[BaseModel] = MetricEvaluation,
    ):
        """Evaluate all metrics for a QA pair"""

        # Get individual metric evaluations
        faithfulness_eval = self.evaluate_faithfulness(question, answer, context)
        relevancy_eval = self.evaluate_answer_relevancy(question, answer)
        precision_eval = self.evaluate_context_precision(question, context)
        completeness_eval = self.evaluate_completeness(question, answer, context)
        hallucination_eval = self.evaluate_hallucination(question, answer, context)

        # Final comprehensive evaluation
        final_prompt = f"""
        # Comprehensive RAG Evaluation

        ## Task
        Based on the individual metric evaluations below, provide a comprehensive evaluation with specific scores for each metric.

        ## Individual Evaluations

        ### Faithfulness
        {faithfulness_eval}

        ### Answer Relevancy
        {relevancy_eval}

        ### Context Precision
        {precision_eval}

        ### Completeness
        {completeness_eval}

        ### Hallucination Detection
        {hallucination_eval}

        ## Output Format
        Provide your evaluation in the following JSON format:
        {response_model(
            overall_score=0.85,
            individual_scores=[
                MetricScore(metric_name="faithfulness", score=0.9, explanation="Explanation here", evidence="Evidence here"),
                MetricScore(metric_name="answer_relevancy", score=0.8, explanation="Explanation here", evidence="Evidence here"),
                MetricScore(metric_name="context_precision", score=0.7, explanation="Explanation here", evidence="Evidence here"),
                MetricScore(metric_name="completeness", score=0.9, explanation="Explanation here", evidence="Evidence here"),
                MetricScore(metric_name="hallucination_score", score=0.1, explanation="Lower is better - explanation here", evidence="Evidence here"),
            ],
            summary="Overall evaluation summary"
        ).model_dump_json()}

        ## Calculation Rules
        - Overall score = (faithfulness + answer_relevancy + context_precision + completeness + (1 - hallucination_score)) / 5
        - Provide specific scores between 0.0 and 1.0 for each metric
        - Include detailed explanations and evidence for each score
        """

        return self.client.generate_text(final_prompt, response_json=response_json)
