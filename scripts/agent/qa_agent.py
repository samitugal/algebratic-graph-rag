from pydantic import BaseModel, Field

from scripts.model.language_models.openai_client import OpenAIClient
from scripts.plugins.graph_searcher import GraphSearcher


class QAResponse(BaseModel):
    answer: str = Field(description="Your detailed answer to the question")


class QAAgent:
    def __init__(self, client: OpenAIClient, graph_searcher: GraphSearcher):
        self.client = client
        self.graph_searcher = graph_searcher

    def invoke(self, prompt: str, response_json: bool = False):
        related_nodes = self.graph_searcher.search_graph_with_embeddings(prompt)
        related_nodes_text = f"\n".join(
            [f"{node['node_id']}: {node['content']}" for node in related_nodes]
        )

        prompt = f"""
        ### Instructions
        You are a helpful assistant that can answer questions about the user request.
        Use the helper informations to answer the question.
        If you don't have enough information, say "I don't know".
        If you have enough information, answer the question.
        You dont have output token limitation, anwsers should be detailed.

        ### Question
        {prompt}

        ### Helper Informations
        {related_nodes_text}

        ### Output Format
        Your output must be in JSON format.
        {QAResponse(answer="your detailed answer here").model_dump_json()}
        """
        return self.client.generate_text(prompt, response_json=response_json)
