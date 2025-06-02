import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client_name = "OpenAI"
        if self.api_key is None:
            raise ValueError(f"{self.client_name} API key is not set")

        self.client = OpenAI(api_key=self.api_key)

    def generate_text(
        self, prompt: str, model: str = "gpt-3.5-turbo", response_json: bool = False
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if response_json else None,
        )
        return response.choices[0].message.content


class OpenRouterClient(OpenAIClient):
    def __init__(self, api_key: str = None):
        self.client_name = "OpenRouter"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        super().__init__(api_key)
