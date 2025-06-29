from scripts.graphdb.ne4oj_client import Neo4jClient
from scripts.model.embeddings import EmbeddingGenerator


class GraphSearcher:
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.embedding_generator = EmbeddingGenerator()

    def search_graph_with_knn(self, query: str, k: int = 3):
        """Search the graph with KNN"""
        query_embedding = self.embedding_generator.generate_embeddings_from_text(query)
        knn_results = self.db_client.knn_search(query_embedding, k)

        return [
            {
                "node_id": result["id"],
                "content": result["text"],
                "score": result["score"],
            }
            for result in knn_results
        ]

    def search_graph_with_embeddings(self, query: str, k_hops: int = 3, top_k: int = 6):
        """PageRank-based search with optimized single-seed strategy"""

        query_embedding = self.embedding_generator.generate_embeddings_from_text(query)

        # Optimized single-seed strategy with quality boost
        target_nodes = self.db_client.knn_search(query_embedding, k=1)

        if not target_nodes:
            print("No target nodes found!")
            return []

        start_node_id = target_nodes[0]["id"]

        traversal_nodes = self.db_client.k_hop_traversal_with_content(
            start_node_id, k_hops
        )

        if not traversal_nodes:
            print("No nodes found in traversal!")
            return []

        enriched_nodes = []
        for node in traversal_nodes:
            if "embedding" in node and node["embedding"]:
                node_embedding = node["embedding"]
                similarity = self.embedding_generator.calculate_similarity(
                    query_embedding, node_embedding
                )

                hop_distance = node.get("hop_distance", 1)
                path_weight = node.get("path_weight", 1.0)

                # Simplified enhanced scoring: Focus on quality over complexity
                quality_boost = 1.0 + min(similarity * 0.3, 0.3)  # Max 30% boost
                final_score = similarity * path_weight * quality_boost

                enriched_nodes.append(
                    {
                        "node_id": node["node_id"],
                        "content": node["content"],
                        "query_similarity": similarity,
                        "hop_distance": hop_distance,
                        "path_weight": path_weight,
                        "pagerank_score": final_score,
                        "quality_boost": quality_boost,
                        "relationship_type": node.get("relationship_type", "unknown"),
                    }
                )

        # Conservative re-ranking: Only boost very high similarity nodes
        for node in enriched_nodes:
            if node["query_similarity"] > 0.75:  # Higher threshold for boost
                node["pagerank_score"] *= 1.1

        enriched_nodes.sort(key=lambda x: x["pagerank_score"], reverse=True)
        top_nodes = enriched_nodes[:top_k]

        return top_nodes
