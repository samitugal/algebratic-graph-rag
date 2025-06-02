from dotenv import load_dotenv
import glob
import numpy as np
import os
import nltk
import re
import string
import torch
from collections import defaultdict

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models_cache"

import warnings
warnings.filterwarnings("ignore")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from scripts.model.language_models import SimpleSummarizer
load_dotenv()

class EmbeddingGenerator:
    _models = {}
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model_name = embedding_model_name
        self.data_dir = os.environ.get("DATA_ABS_DIR")
        self.device = "cpu"
        
        if embedding_model_name not in self._models:
            model = SentenceTransformer(embedding_model_name)
            model.max_seq_length = 512
            self._models[embedding_model_name] = model
        
        self.model = self._models[embedding_model_name]

    def _get_data(self) -> str:
        txt_files = glob.glob(os.path.join(self.data_dir, "**/*.txt"), recursive=True)
        data = []
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data.append(f.read())
        return " \n".join(data)
    
    def _clean_text(self, text: str) -> str:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
        return text

    def _chunk_text(self, max_chars=1200, overlap=250):
        text = self._clean_text(self._get_data())
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

    def generate_embeddings_from_chunks(self, chunks: list[str]):
        embeddings = self.model.encode(
            chunks, 
            show_progress_bar=False, 
            batch_size=16, 
            device=self.device, 
            normalize_embeddings=True,
            convert_to_tensor=False
        )
        return embeddings
    
    def generate_embeddings_from_text(self, text: str):
        embeddings = self.model.encode(
            text, 
            show_progress_bar=False, 
            batch_size=16, 
            device=self.device,
            normalize_embeddings=True,
            convert_to_tensor=False
        )
        return embeddings

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

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
        self.embeddings_generator = EmbeddingGenerator()

    def summarize(self, cluster_ids):
        selected = [self.chunks[i] for i in cluster_ids]
        full_text = " ".join(selected)
        embeddings = self.embeddings_generator.generate_embeddings_from_text(full_text)
        return {
            "id": cluster_ids[0],
            "content": self.summarizer.summarize(full_text),
            "embedding": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        }
    

class HierarchicalSummarizer:
    def __init__(self, summarizer: SimpleSummarizer, embedding_generator: EmbeddingGenerator):
        self.summarizer = summarizer
        self.embedding_generator = embedding_generator
        self.all_summaries = []
        self.parent_child_relations = []

    def create_hierarchical_summaries(self, initial_summaries: list[dict], similarity_threshold: float = 0.75):
        current_level = 1
        current_summaries = initial_summaries.copy()
        
        for summary in current_summaries:
            summary["level"] = current_level
            summary["original_id"] = summary["id"]
            summary["id"] = f"L{current_level}_S{summary['id']}"
        
        self.all_summaries.extend(current_summaries)
        
        while len(current_summaries) > 1:
            print(f"\n--- Level {current_level} processing: {len(current_summaries)} summaries ---")
            
            embeddings = [summary["embedding"] for summary in current_summaries]
            
            linker = ChunkLinker(embeddings)
            links = linker.get_chunk_links(threshold=similarity_threshold)
            
            if not links:
                print("No similar summaries found, stopping recursion")
                break
                
            builder = SubgraphBuilder(links)
            clusters = builder.build_subgraphs()
            clusters = linker.add_isolated_chunks(clusters, len(current_summaries))
            
            print(f"Created {len(clusters)} clusters for next level")
            
            if len(clusters) >= len(current_summaries):
                print("No reduction in clusters, stopping recursion")
                break
            
            next_level_summaries = []
            current_level += 1
            
            for cluster_idx, cluster in enumerate(clusters):
                cluster_contents = [current_summaries[i]["content"] for i in cluster]
                combined_text = " ".join(cluster_contents)
                
                new_summary_content = self.summarizer.summarize(combined_text)
                new_embedding = self.embedding_generator.generate_embeddings_from_text(new_summary_content)
                
                new_summary = {
                    "id": f"L{current_level}_S{cluster_idx}",
                    "content": new_summary_content,
                    "embedding": new_embedding.tolist() if hasattr(new_embedding, 'tolist') else new_embedding,
                    "level": current_level,
                    "cluster": cluster
                }
                
                next_level_summaries.append(new_summary)
                
                for child_idx in cluster:
                    self.parent_child_relations.append({
                        "parent_id": new_summary["id"],
                        "child_id": current_summaries[child_idx]["id"]
                    })
            
            self.all_summaries.extend(next_level_summaries)
            current_summaries = next_level_summaries
        
        if current_summaries:
            root_content = " ".join([s["content"] for s in current_summaries])
            final_summary = self.summarizer.summarize(root_content)
            root_embedding = self.embedding_generator.generate_embeddings_from_text(final_summary)
            
            root_summary = {
                "id": "ROOT",
                "content": final_summary,
                "embedding": root_embedding.tolist() if hasattr(root_embedding, 'tolist') else root_embedding
            }
            
            root_connections = [s["id"] for s in current_summaries]
            
            return root_summary, root_connections
        
        return None, []



