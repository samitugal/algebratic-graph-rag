import os
import warnings

warnings.filterwarnings("ignore")

from scripts.graph_generator import generate_graph
from scripts.graph_searcher import GraphSearcher
from scripts.graphdb.ne4oj_client import Neo4jClient


db_config = {
    "uri": os.getenv("NEO4J_URI") or "bolt://localhost:7687",
    "user": os.getenv("NEO4J_USERNAME") or "neo4j",
    "password": os.getenv("NEO4J_PASSWORD") or "password",
}

db_client = Neo4jClient(**db_config)

if __name__ == "__main__":
    print("\nSearching graph...")
    graph_searcher = GraphSearcher(db_client)

    results = graph_searcher.search_graph_with_embeddings(
        "İzmir tarihi hakkında bilgi", k_hops=2
    )

    if results:
        print("\n" + "=" * 50)
        print("CALCULATING PAGERANK SCORES")
        print("=" * 50)
        ranked_results = graph_searcher.calculate_pagerank_scores(results)

        print(f"\n" + "=" * 50)
        print("FINAL RESULTS (Query + PageRank Combined)")
        print("=" * 50)
        for i, node in enumerate(ranked_results[:5]):
            print(f"\n{i+1}. Node ID: {node['node_id']}")
            print(f"   PageRank Score: {node['pagerank_score']:.4f}")
            print(f"   Query Similarity: {node['query_similarity']:.4f}")
            print(
                f"   Combined Score: {node['pagerank_score'] * node['query_similarity']:.4f}"
            )
            print(f"   Hop Distance: {node['hop_distance']}")
            print(f"   Content: {node['content'][:150]}...")
    else:
        print("No results found!")
