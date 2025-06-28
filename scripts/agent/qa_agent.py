from pydantic import BaseModel, Field

from scripts.agent.base_agent import BaseAgent
from scripts.model.language_models.openai_client import OpenAIClient
from scripts.plugins.graph_searcher import GraphSearcher


class QAResponse(BaseModel):
    answer: str = Field(description="Your detailed answer to the question")


class QAAgent(BaseAgent):
    def __init__(self, client: OpenAIClient):
        super().__init__(client)

    def invoke(
        self,
        user_request: str,
        additional_context: str = "",
        response_json: bool = False,
        response_model: type[BaseModel] = QAResponse,
    ):
        prompt = f"""
        ### Instructions
        You are a helpful assistant that can answer questions about the user request.
        Use the helper informations to answer the question.
        If you don't have enough information, say "I don't know".
        If you have enough information, answer the question.
        You dont have output token limitation, anwsers should be detailed.

        ### Question
        {user_request}

        ### Helper Informations
        {additional_context}

        ### Output Format
        Your output must be in JSON format.
        {response_model(answer="your detailed answer here").model_dump_json()}
        """
        return self.client.generate_text(prompt, response_json=response_json)
