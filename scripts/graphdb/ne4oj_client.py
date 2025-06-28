# pylint: disable= all
from neo4j import GraphDatabase


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clean_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_chunk_nodes(self, chunks: list[dict]):
        with self.driver.session() as session:
            for chunk in chunks:
                session.run(
                    """
                CREATE (:Chunk {id: $id, content: $content, embedding: $embedding})
                """,
                    {
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "embedding": chunk["embedding"],
                    },
                )

        # Vector index oluştur
        self.create_vector_index()

    def create_vector_index(self):
        """Create vector index for chunk embeddings"""
        with self.driver.session() as session:
            try:
                session.run(
                    """
                    CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """
                )
                print("Vector index 'chunkEmbedding' created successfully")
            except Exception as e:
                print(f"Vector index creation error: {e}")

    def create_chunk_edges(self, links: list[dict]):
        with self.driver.session() as session:
            for link in links:
                session.run(
                    """
                MATCH (c1:Chunk {id: $source}), (c2:Chunk {id: $target})
                MERGE (c1)-[:RELATED_TO {weight: $weight, similarity: $similarity}]->(c2)
                """,
                    {
                        "source": link["source"],
                        "target": link["target"],
                        "weight": link["similarity"],
                        "similarity": link["similarity"],
                    },
                )

    def create_summary_nodes(self, summaries: list[dict]):
        with self.driver.session() as session:
            for summary in summaries:
                session.run(
                    """
                CREATE (s:Summary {id: $id, content: $content, embedding: $embedding})
                """,
                    {
                        "id": summary["id"],
                        "content": summary["content"],
                        "embedding": summary["embedding"],
                    },
                )

    def create_summary_edges_for_chunk(self, summaries: list[dict]):
        with self.driver.session() as session:
            for summary in summaries:
                summary_id = summary["id"]
                weight = summary.get("weight", 0.8)  # Default weight if not provided
                for chunk_id in summary["cluster"]:
                    session.run(
                        """
                    MATCH (c:Chunk {id: $chunk_id}), (s:Summary {id: $summary_id})
                    MERGE (c)-[:SUMMARIZED_IN {weight: $weight, relationship_type: 'summarization'}]->(s)
                    """,
                        {
                            "chunk_id": chunk_id,
                            "summary_id": summary_id,
                            "weight": weight,
                        },
                    )

    def create_summary_similarity_edges(self, summary_links: list[dict]):
        with self.driver.session() as session:
            for link in summary_links:
                session.run(
                    """
                MATCH (s1:Summary {id: $source}), (s2:Summary {id: $target})
                MERGE (s1)-[:SIMILAR_TO {weight: $weight, similarity: $similarity}]->(s2)
                """,
                    {
                        "source": link["source"],
                        "target": link["target"],
                        "weight": link["similarity"],
                        "similarity": link["similarity"],
                    },
                )

    def create_hierarchical_summary_nodes(self, summaries: list[dict], level: int):
        with self.driver.session() as session:
            for summary in summaries:
                session.run(
                    """
                CREATE (s:Summary {id: $id, content: $content, embedding: $embedding, level: $level})
                """,
                    {
                        "id": summary["id"],
                        "content": summary["content"],
                        "embedding": summary["embedding"],
                        "level": level,
                    },
                )

    def create_hierarchical_edges(self, parent_child_pairs: list[dict]):
        with self.driver.session() as session:
            for pair in parent_child_pairs:
                session.run(
                    """
                MATCH (parent:Summary {id: $parent_id}), (child:Summary {id: $child_id})
                MERGE (child)-[:SUMMARIZED_BY {weight: 1.0, relationship_type: 'hierarchical'}]->(parent)
                """,
                    {"parent_id": pair["parent_id"], "child_id": pair["child_id"]},
                )

    def create_root_node(self, root_summary: dict):
        with self.driver.session() as session:
            session.run(
                """
            CREATE (r:RootSummary {id: $id, content: $content, embedding: $embedding})
            """,
                {
                    "id": root_summary["id"],
                    "content": root_summary["content"],
                    "embedding": root_summary["embedding"],
                },
            )

    def connect_to_root(self, summary_ids: list[int], root_id: str):
        with self.driver.session() as session:
            for summary_id in summary_ids:
                session.run(
                    """
                MATCH (s:Summary {id: $summary_id}), (r:RootSummary {id: $root_id})
                MERGE (s)-[:CONNECTS_TO_ROOT {weight: 1.0, relationship_type: 'root_connection'}]->(r)
                """,
                    {"summary_id": summary_id, "root_id": root_id},
                )

    def get_weighted_connections(self, node_id: str, min_weight: float = 0.0):
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH (n {id: $node_id})-[r]->(connected)
            WHERE r.weight >= $min_weight
            RETURN connected.id as connected_id, r.weight as weight, type(r) as relationship_type
            ORDER BY r.weight DESC
            """,
                {"node_id": node_id, "min_weight": min_weight},
            )
            return [
                {
                    "connected_id": record["connected_id"],
                    "weight": record["weight"],
                    "relationship_type": record["relationship_type"],
                }
                for record in result
            ]

    def get_high_similarity_clusters(self, min_weight: float = 0.8):
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH (n1)-[r:RELATED_TO|SIMILAR_TO]->(n2)
            WHERE r.weight >= $min_weight
            RETURN n1.id as node1, n2.id as node2, r.weight as weight, type(r) as relationship_type
            ORDER BY r.weight DESC
            """,
                {"min_weight": min_weight},
            )
            return [
                {
                    "node1": record["node1"],
                    "node2": record["node2"],
                    "weight": record["weight"],
                    "relationship_type": record["relationship_type"],
                }
                for record in result
            ]

    def calculate_node_centrality(self):
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH (n)-[r]->()
            WITH n, sum(r.weight) as total_outgoing_weight, count(r) as outgoing_count
            MATCH ()-[r2]->(n)
            WITH n, total_outgoing_weight, outgoing_count, sum(r2.weight) as total_incoming_weight, count(r2) as incoming_count
            RETURN n.id as node_id, 
                   total_outgoing_weight, outgoing_count,
                   total_incoming_weight, incoming_count,
                   (total_outgoing_weight + total_incoming_weight) as total_weight
            ORDER BY total_weight DESC
            """
            )
            return [
                {
                    "node_id": record["node_id"],
                    "total_weight": record["total_weight"],
                    "outgoing_weight": record["total_outgoing_weight"],
                    "incoming_weight": record["total_incoming_weight"],
                    "total_connections": record["outgoing_count"]
                    + record["incoming_count"],
                }
                for record in result
            ]

    def find_weighted_path(self, start_id: str, end_id: str, max_hops: int = 5):
        """İki node arasında ağırlıklı en iyi yolu bul"""
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH path = (start {id: $start_id})-[*1..$max_hops]-(end {id: $end_id})
            WITH path, relationships(path) as rels
            WITH path, reduce(total = 0, r in rels | total + r.weight) as path_weight
            RETURN path, path_weight
            ORDER BY path_weight DESC
            LIMIT 1
            """,
                {"start_id": start_id, "end_id": end_id, "max_hops": max_hops},
            )

            record = result.single()
            if record:
                return {
                    "path_weight": record["path_weight"],
                    "path_length": len(record["path"].relationships),
                }
            return None

    def get_similar_nodes(self, node_id: str, min_weight: float = 0.7, limit: int = 10):
        """Bir node'a benzer node'ları ağırlığa göre sırala"""
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH (n {id: $node_id})-[r:RELATED_TO|SIMILAR_TO]-(similar)
            WHERE r.weight >= $min_weight
            RETURN similar.id as similar_id, similar.content as content, r.weight as similarity
            ORDER BY r.weight DESC
            LIMIT $limit
            """,
                {"node_id": node_id, "min_weight": min_weight, "limit": limit},
            )
            return [
                {
                    "similar_id": record["similar_id"],
                    "content": (
                        record["content"][:100] + "..."
                        if len(record["content"]) > 100
                        else record["content"]
                    ),
                    "similarity": record["similarity"],
                }
                for record in result
            ]

    def get_weighted_subgraph(
        self, node_id: str, min_weight: float = 0.8, max_depth: int = 2
    ):
        """Bir node etrafında ağırlıklı alt-graph oluştur"""
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH (center {id: $node_id})-[r*1..$max_depth]-(connected)
            WHERE ALL(rel in r WHERE rel.weight >= $min_weight)
            WITH center, connected, r
            RETURN center.id as center_id, 
                   collect(DISTINCT connected.id) as connected_nodes,
                   reduce(total = 0, rel in r | total + rel.weight) as total_path_weight
            """,
                {"node_id": node_id, "min_weight": min_weight, "max_depth": max_depth},
            )

            record = result.single()
            if record:
                return {
                    "center_id": record["center_id"],
                    "connected_nodes": record["connected_nodes"],
                    "total_path_weight": record["total_path_weight"],
                }
            return None

    def knn_search(self, query_embedding, k: int = 3):
        """KNN search for the query embedding"""

        if hasattr(query_embedding, "tolist"):
            embedding = query_embedding.tolist()
        else:
            embedding = query_embedding

        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes(
                'chunkEmbedding',
                $k,
                $embedding
                ) YIELD node, score
                RETURN node.id AS id, node.content AS text, score
                ORDER BY score ASC
                """,
                k=k,
                embedding=embedding,
            )
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "score": record["score"],
                }
                for record in result
            ]

    def k_hop_traversal(self, node_id: str, k: int = 3):
        """K-hop traversal for the node"""
        with self.driver.session() as session:
            result = session.run(
                """
            MATCH (n {id: $node_id})-[r*1..$k]-(connected)
            RETURN connected.id as connected_id, r.weight as weight, type(r) as relationship_type
            ORDER BY r.weight DESC
            """,
                {"node_id": node_id, "k": k},
            )
            return [
                {
                    "connected_id": record["connected_id"],
                    "weight": record["weight"],
                    "relationship_type": record["relationship_type"],
                }
                for record in result
            ]

    def k_hop_traversal_with_content(self, node_id: str, k_hops: int = 2):
        """K-hop traversal with node content and embeddings"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start {id: $node_id})-[r*1..5]-(connected)
                WITH connected, relationships(path) as rels, length(path) as hop_distance
                WHERE hop_distance <= $k_hops AND connected.id <> $node_id
                WITH connected.id as node_id, connected.content as content, connected.embedding as embedding,
                     hop_distance, 
                     reduce(total = 1.0, rel in rels | total * coalesce(rel.weight, 1.0)) as path_weight_product,
                     head([rel in rels | type(rel)]) as relationship_type
                WITH node_id, content, embedding, 
                     min(hop_distance) as min_hop_distance,
                     max(path_weight_product) as best_weight,
                     collect(relationship_type)[0] as relationship_type
                RETURN DISTINCT 
                    node_id,
                    content,
                    embedding,
                    min_hop_distance as hop_distance,
                    best_weight as path_weight,
                    relationship_type
                ORDER BY min_hop_distance ASC, best_weight DESC
                """,
                {"node_id": node_id, "k_hops": k_hops},
            )
            return [
                {
                    "node_id": record["node_id"],
                    "content": record["content"],
                    "embedding": record["embedding"],
                    "hop_distance": record["hop_distance"],
                    "path_weight": record["path_weight"] or 1.0,
                    "relationship_type": record["relationship_type"],
                }
                for record in result
            ]
