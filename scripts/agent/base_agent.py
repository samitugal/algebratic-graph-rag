from abc import ABC, abstractmethod
from pydantic import BaseModel

from scripts.model.language_models.openai_client import OpenAIClient


class BaseAgent:
    def __init__(self, client: OpenAIClient):
        self.client = client

    @abstractmethod
    def invoke(
        self,
        user_request: str,
        response_model: BaseModel,
        response_json: bool = True,
    ):
        pass
