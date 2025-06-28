import os
from pyexpat import model
import warnings

warnings.filterwarnings("ignore")

from scripts.agent.qa_agent import QAAgent
from scripts.graph_generator import generate_graph
from scripts.plugins.graph_searcher import GraphSearcher
from scripts.graphdb.ne4oj_client import Neo4jClient
from scripts.model.language_models.openai_client import OpenRouterClient

db_config = {
    "uri": os.getenv("NEO4J_URI") or "bolt://localhost:7687",
    "user": os.getenv("NEO4J_USERNAME") or "neo4j",
    "password": os.getenv("NEO4J_PASSWORD") or "password",
}

db_client = Neo4jClient(**db_config)

client = OpenRouterClient(model="openai/gpt-4o-mini")
graph_searcher = GraphSearcher(db_client)
agent = QAAgent(client, graph_searcher)

if __name__ == "__main__":
    response = agent.invoke("İzmir tarihi hakkında bilgi")
    print(response)
