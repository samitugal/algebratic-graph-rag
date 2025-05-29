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
                session.run("""
                CREATE (:Chunk {id: $id, content: $content, embedding: $embedding})
                """, {"id": chunk["id"], "content": chunk["content"], "embedding": chunk["embedding"]})
    
    def create_chunk_edges(self, links: list[dict]):
        with self.driver.session() as session:
            for link in links:
                session.run("""
                MATCH (c1:Chunk {id: $source}), (c2:Chunk {id: $target})
                MERGE (c1)-[:RELATED_TO {score: $similarity}]->(c2)
                """, {"source": link["source"], "target": link["target"], "similarity": link["similarity"]})
