from scripts.graphdb.ne4oj_client import Neo4jClient
from scripts.model.embeddings import EmbeddingGenerator


class GraphSearcher:
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.embedding_generator = EmbeddingGenerator()

    def search_graph_with_embeddings(self, query: str, k_hops: int = 2):
        """Search the graph with embeddings using k-hop traversal"""
        print(f"Searching for: {query}")

        # Query'yi embedding'e çevir
        query_embedding = self.embedding_generator.generate_embeddings_from_text(query)

        # KNN ile en yakın 1 node'u bul
        target_nodes = self.db_client.knn_search(query_embedding, k=1)

        if not target_nodes:
            print("No target node found!")
            return []

        start_node_id = target_nodes[0]["id"]
        print(
            f"Starting node: {start_node_id} \n Starting node content: {target_nodes[0]['text']}"
        )

        # 2 dereceli k-hop traversal yap
        traversal_nodes = self.db_client.k_hop_traversal_with_content(
            start_node_id, k_hops
        )

        if not traversal_nodes:
            print("No nodes found in traversal!")
            return []

        # Her node için query ile benzerlik hesapla
        enriched_nodes = []
        for node in traversal_nodes:
            if "embedding" in node and node["embedding"]:
                # Node embedding'i ile query embedding'i arasındaki benzerlik
                node_embedding = node["embedding"]
                similarity = self.embedding_generator.calculate_similarity(
                    query_embedding, node_embedding
                )

                enriched_nodes.append(
                    {
                        "node_id": node["node_id"],
                        "content": node["content"],
                        "original_weight": node.get("weight", 0.0),
                        "query_similarity": similarity,
                        "relationship_type": node.get("relationship_type", "unknown"),
                        "hop_distance": node.get("hop_distance", 1),
                    }
                )

        # Benzerlik skoruna göre sırala
        enriched_nodes.sort(key=lambda x: x["query_similarity"], reverse=True)

        print(f"Found {len(enriched_nodes)} nodes in {k_hops}-hop traversal:")
        for i, node in enumerate(enriched_nodes[:10]):  # İlk 10'unu göster
            print(f"{i+1}. Node ID: {node['node_id']}")
            print(f"   Query Similarity: {node['query_similarity']:.4f}")
            print(f"   Original Weight: {node['original_weight']:.4f}")
            print(f"   Hop Distance: {node['hop_distance']}")
            print(f"   Content: {node['content'][:100]}...")
            print()

        return enriched_nodes

    def calculate_pagerank_scores(
        self, nodes, damping_factor=0.85, max_iterations=100, tolerance=1e-6
    ):
        """PageRank algoritması - node'ların önem skorlarını hesapla"""
        if not nodes:
            return []

        n = len(nodes)
        node_ids = [node["node_id"] for node in nodes]

        # Başlangıç skorları (eşit dağıtım)
        scores = {node_id: 1.0 / n for node_id in node_ids}

        # Node'lar arası bağlantı matrisi oluştur (query similarity'e göre)
        links = {}
        for i, node in enumerate(nodes):
            links[node["node_id"]] = []
            for j, other_node in enumerate(nodes):
                if i != j:
                    # Query similarity + hop distance göz önünde bulundur
                    link_weight = (
                        node["query_similarity"]
                        * other_node["query_similarity"]
                        * (
                            1.0
                            / max(
                                1,
                                abs(node["hop_distance"] - other_node["hop_distance"]),
                            )
                        )
                    )
                    if link_weight > 0.1:  # Threshold
                        links[node["node_id"]].append(
                            (other_node["node_id"], link_weight)
                        )

        # PageRank iterasyonları
        for iteration in range(max_iterations):
            new_scores = {}

            for node_id in node_ids:
                # Temel skor
                new_score = (1 - damping_factor) / n

                # Diğer node'lardan gelen linkler
                incoming_score = 0
                for other_id in node_ids:
                    if other_id != node_id and other_id in links:
                        for linked_id, weight in links[other_id]:
                            if linked_id == node_id:
                                # Gelen link'in ağırlığı
                                outgoing_count = len(links[other_id])
                                if outgoing_count > 0:
                                    incoming_score += (
                                        scores[other_id] * weight
                                    ) / outgoing_count

                new_scores[node_id] = new_score + damping_factor * incoming_score

            # Convergence kontrolü
            diff = sum(
                abs(new_scores[node_id] - scores[node_id]) for node_id in node_ids
            )
            scores = new_scores

            if diff < tolerance:
                print(f"PageRank converged after {iteration + 1} iterations")
                break

        # Skorları node'lara ekle
        for node in nodes:
            node["pagerank_score"] = scores[node["node_id"]]

        # PageRank skoruna göre sırala
        nodes.sort(key=lambda x: x["pagerank_score"], reverse=True)

        print(f"\nPageRank Scores (Top 10):")
        for i, node in enumerate(nodes[:10]):
            print(
                f"{i+1}. Node {node['node_id']}: PageRank={node['pagerank_score']:.4f}, "
                f"Query Similarity={node['query_similarity']:.4f}"
            )

        return nodes
