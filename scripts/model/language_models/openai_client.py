import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client_name = "OpenAI"
        self.model = model
        if self.api_key is None:
            raise ValueError(f"{self.client_name} API key is not set")

        self.client = OpenAI(api_key=self.api_key)

    def generate_text(
        self, prompt: str, model: Optional[str] = None, response_json: bool = False
    ) -> str:
        kwargs = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class OpenRouterClient(OpenAIClient):
    def __init__(
        self, api_key: Optional[str] = None, model: str = "openai/gpt-4o-mini"
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        )
        self.client_name = "OpenRouter"
        self.model = model

        if self.api_key is None:
            raise ValueError(f"{self.client_name} API key is not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
