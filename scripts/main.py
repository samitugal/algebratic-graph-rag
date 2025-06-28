import os
from pyexpat import model
import warnings

warnings.filterwarnings("ignore")

from scripts.agent import JudgeAgent, QAAgent
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
qa_agent = QAAgent(client)
judge_agent = JudgeAgent(client)


def main():
    topic = "Explain the computer programming and artificial intelligence"
    print("#" * 100)
    print("Pagerank-based GraphRAG")
    related_nodes = graph_searcher.search_graph_with_embeddings(
        topic, k_hops=2, top_k=10
    )
    additional_context = "\n".join(
        [f"{node['node_id']}: {node['content']}" for node in related_nodes]
    )
    pagerank_response = qa_agent.invoke(topic, additional_context)
    print(pagerank_response)
    print("#" * 100)

    print("KNN-based GraphRAG")
    related_nodes = graph_searcher.search_graph_with_knn(topic, k=10)
    additional_context = "\n".join(
        [f"{node['node_id']}: {node['content']}" for node in related_nodes]
    )
    knn_response = qa_agent.invoke(topic, additional_context)
    print(knn_response)
    print("#" * 100)

    print("GraphRAG-based")
    related_nodes = graph_searcher.search_graph_with_knn(topic, k=1)
    additional_context = "\n".join(
        [f"{node['node_id']}: {node['content']}" for node in related_nodes]
    )
    graphrag_response = qa_agent.invoke(topic, additional_context)
    print(graphrag_response)
    print("#" * 100)

    judge_request = f"""
    ### Topic
    {topic}

    ### Pagerank-based GraphRAG
    {pagerank_response}

    ### KNN-based GraphRAG
    {knn_response}

    ### GraphRAG-based
    {graphrag_response}
    """

    judge_response = judge_agent.invoke(judge_request)
    print(judge_response)
    print("#" * 100)


if __name__ == "__main__":
    main()
