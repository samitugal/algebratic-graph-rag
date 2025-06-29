import warnings

warnings.filterwarnings("ignore")

import os

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dotenv import load_dotenv

from scripts.graphdb.ne4oj_client import Neo4jClient
from scripts.model.embeddings import (
    ChunkLinker,
    EmbeddingGenerator,
    HierarchicalSummarizer,
    SubgraphBuilder,
    SubgraphSummarizer,
)
from scripts.model.language_models import SimpleSummarizer

load_dotenv()

db_config = {
    "uri": os.getenv("NEO4J_URI") or "bolt://localhost:7687",
    "user": os.getenv("NEO4J_USERNAME") or "neo4j",
    "password": os.getenv("NEO4J_PASSWORD") or "password",
}

db_client = Neo4jClient(**db_config)


def generate_graph(
    embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
):
    generator = EmbeddingGenerator(embedding_model_name)
    chunks = generator._chunk_text()
    embeddings = generator.generate_embeddings_from_chunks(chunks)

    chunk_objects = [
        {"id": i, "content": chunks[i], "embedding": embeddings[i].tolist()}
        for i in range(len(chunks))
    ]

    linker = ChunkLinker(embeddings)
    links = linker.get_chunk_links()

    link_objects = [
        {"source": link[0], "target": link[1], "similarity": link[2]} for link in links
    ]

    db_client.clean_database()
    db_client.create_chunk_nodes(chunk_objects)
    db_client.create_chunk_edges(link_objects)

    builder = SubgraphBuilder(links)
    subgraph_clusters = builder.build_subgraphs()
    subgraph_clusters = linker.add_isolated_chunks(subgraph_clusters, len(chunks))

    # Create initial summaries (Level 1)
    summarizer_model = SimpleSummarizer()
    summarizer = SubgraphSummarizer(chunks, summarizer_model)

    initial_summaries = []
    for i, cluster in enumerate(subgraph_clusters):
        summary_dict = summarizer.summarize(cluster)
        summary_dict["id"] = i
        summary_dict["cluster"] = cluster
        initial_summaries.append(summary_dict)

    # Create hierarchical summaries
    hierarchical_summarizer = HierarchicalSummarizer(summarizer_model, generator)
    root_summary, root_connections = (
        hierarchical_summarizer.create_hierarchical_summaries(initial_summaries)
    )

    print(f"\n=== Hierarchical Summarization Complete ===")
    print(
        f"Total summary levels created: {len(set(s['level'] for s in hierarchical_summarizer.all_summaries))}"
    )
    print(f"Total summaries: {len(hierarchical_summarizer.all_summaries)}")
    print(
        f"Parent-child relations: {len(hierarchical_summarizer.parent_child_relations)}"
    )

    # Save all hierarchical summaries to database
    for level in sorted(set(s["level"] for s in hierarchical_summarizer.all_summaries)):
        level_summaries = [
            s for s in hierarchical_summarizer.all_summaries if s["level"] == level
        ]
        db_client.create_hierarchical_summary_nodes(level_summaries, level)
        print(f"Created {len(level_summaries)} summaries for level {level}")

    # Create similarity edges between summaries at each level
    for level in sorted(set(s["level"] for s in hierarchical_summarizer.all_summaries)):
        level_summaries = [
            s for s in hierarchical_summarizer.all_summaries if s["level"] == level
        ]
        if len(level_summaries) > 1:
            # Convert embeddings back to numpy arrays for ChunkLinker
            level_embeddings = []
            for s in level_summaries:
                embedding = s["embedding"]
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                level_embeddings.append(embedding)

            level_linker = ChunkLinker(level_embeddings)
            level_links = level_linker.get_chunk_links()

            summary_link_objects = []
            for link in level_links:
                summary_link_objects.append(
                    {
                        "source": level_summaries[link[0]]["id"],
                        "target": level_summaries[link[1]]["id"],
                        "similarity": link[2],
                    }
                )

            if summary_link_objects:
                db_client.create_summary_similarity_edges(summary_link_objects)
                print(
                    f"Created {len(summary_link_objects)} similarity edges for level {level}"
                )
            else:
                print(f"No similarity edges found for level {level} (threshold=0.65)")

    # Create hierarchical edges (parent-child relationships)
    if hierarchical_summarizer.parent_child_relations:
        db_client.create_hierarchical_edges(
            hierarchical_summarizer.parent_child_relations
        )
        print(
            f"Created {len(hierarchical_summarizer.parent_child_relations)} hierarchical edges"
        )

    # Create and connect root node
    if root_summary:
        db_client.create_root_node(root_summary)
        db_client.connect_to_root(root_connections, root_summary["id"])
        print(
            f"Created root node and connected {len(root_connections)} top-level summaries"
        )

    level_1_summaries = [
        s for s in hierarchical_summarizer.all_summaries if s["level"] == 1
    ]
    for summary in level_1_summaries:
        cluster_chunks = summary["cluster"]
        cluster_embeddings = [embeddings[chunk_id] for chunk_id in cluster_chunks]

        # Convert summary embedding back to numpy array if needed
        summary_embedding = summary["embedding"]
        if isinstance(summary_embedding, list):
            summary_embedding = np.array(summary_embedding)

        similarities = []
        for chunk_embedding in cluster_embeddings:
            similarity = generator.calculate_similarity(
                chunk_embedding, summary_embedding
            )
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.7

        for chunk_id in cluster_chunks:
            db_client.create_summary_edges_for_chunk(
                [{"id": summary["id"], "cluster": [chunk_id], "weight": avg_similarity}]
            )

    print(f"\n=== Weighted Edge Analysis ===")

    high_similarity = db_client.get_high_similarity_clusters(min_weight=0.8)
    print(f"High similarity connections (≥0.8): {len(high_similarity)}")
    for conn in high_similarity[:5]:
        print(
            f"  {conn['node1']} ↔ {conn['node2']}: {conn['weight']:.3f} ({conn['relationship_type']})"
        )

    centrality = db_client.calculate_node_centrality()
    print(f"\nTop 5 most connected nodes by weight:")
    for node in centrality[:5]:
        print(
            f"  {node['node_id']}: total_weight={node['total_weight']:.3f}, connections={node['total_connections']}"
        )

    return {
        "all_summaries": hierarchical_summarizer.all_summaries,
        "root_summary": root_summary,
        "hierarchical_relations": hierarchical_summarizer.parent_child_relations,
        "high_similarity_connections": high_similarity,
        "node_centrality": centrality,
    }


if __name__ == "__main__":
    generate_graph()
