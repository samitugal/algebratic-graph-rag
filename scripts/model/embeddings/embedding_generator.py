from dotenv import load_dotenv
import glob
import numpy as np
import os
import nltk
import torch
from collections import defaultdict

nltk.download("punkt")
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from scripts.model.language_models import SimpleSummarizer
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model_name = embedding_model_name
        self.data_dir = os.environ.get("DATA_ABS_DIR")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_data(self) -> str:
        txt_files = glob.glob(os.path.join(self.data_dir, "**/*.txt"), recursive=True)
        data = []
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data.append(f.read())
        return " \n".join(data)

    def _chunk_text(self, max_chars=800, overlap=100):
        text = self._get_data()
        sentences = sent_tokenize(text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) <= max_chars:
                current += " " + sentence
            else:
                chunks.append(current.strip())
                current = sentence[-overlap:]
        if current:
            chunks.append(current.strip())
        return chunks

    def generate_embeddings(self):
        chunks = self._chunk_text()
        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(chunks, show_progress_bar= True, batch_size= 100, device= self.device, normalize_embeddings= True)
        return embeddings
    

class ChunkLinker:
    def __init__(self, embeddings: list[np.ndarray]):
        self.embeddings = embeddings
    
    def _cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def get_chunk_links(self, threshold: float = 0.75):
        links = []
        for i, embedding in enumerate(self.embeddings):
            for j in range(i + 1, len(self.embeddings)):
                similarity = self._cosine_similarity(embedding, self.embeddings[j])
                if similarity >= threshold:
                    links.append((i, j, similarity))
        return links
    
    def add_isolated_chunks(self, clusters, total_chunks):
        clustered = set(i for cluster in clusters for i in cluster)
        all_chunks = set(range(total_chunks))
        isolated = all_chunks - clustered

        if isolated:
            clusters += [[i] for i in isolated]
        return clusters
    

class SubgraphBuilder:
    def __init__(self, links: list[tuple[int, int, float]]):
        self.links = links

    def build_subgraphs(self):
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        nodes = set()
        for i, j, _ in self.links:
            nodes.update([i, j])
            parent.setdefault(i, i)
            parent.setdefault(j, j)
            union(i, j)

        clusters = defaultdict(list)
        for node in nodes:
            root = find(node)
            clusters[root].append(node)

        return list(clusters.values())
    

class SubgraphSummarizer:
    def __init__(self, chunks: list[str], summarizer: SimpleSummarizer):
        self.chunks = chunks
        self.summarizer = summarizer

    def summarize(self, cluster_ids):
        selected = [self.chunks[i] for i in cluster_ids]
        full_text = " ".join(selected)
        return self.summarizer.summarize(full_text)
    

def embedding_pipeline(embedding_model_name: str = "all-MiniLM-L6-v2"):
    generator = EmbeddingGenerator(embedding_model_name)
    chunks = generator._chunk_text()
    embeddings = generator.generate_embeddings()

    linker = ChunkLinker(embeddings)
    links = linker.get_chunk_links()

    builder = SubgraphBuilder(links)
    subgraph_clusters = builder.build_subgraphs()
    subgraph_clusters = linker.add_isolated_chunks(subgraph_clusters, len(chunks))

    summarizer_model = SimpleSummarizer() 
    summarizer = SubgraphSummarizer(chunks, summarizer_model)

    subgraph_summaries = [
        {"cluster": cluster, "summary": summarizer.summarize(cluster)}
        for cluster in subgraph_clusters
    ]

    return subgraph_summaries



