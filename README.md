# Algebraic Graph RAG

Algebraic Graph RAG is an experimental framework that demonstrates how a knowledge
graph and modern language models can be combined for retrieval‑augmented
generation (RAG). Text documents are embedded, linked by similarity and
summarised hierarchically. The resulting graph is stored in Neo4j and can be
searched to provide context for a language model when answering questions.

## Features

- **Graph Construction** – `scripts/graph_generator.py` reads text files under
  `scripts/data`, creates embeddings with
  `sentence-transformers`, links similar text chunks and builds subgraphs. The
  subgraphs are summarised recursively with a HuggingFace model and the whole
  hierarchy is written to Neo4j, including similarity and hierarchical
  relationships.
- **Graph Search** – `scripts/plugins/graph_searcher.py` provides two search
  strategies:
  - **K‑NN search** using Neo4j's vector index.
  - **PageRank‑based traversal** that expands from the most similar node and
    scores results by similarity and graph connectivity.
- **Agents** –
  - `QAAgent` queries the graph and generates answers using an LLM.
  - `JudgeAgent` compares the outputs of different retrieval strategies.
- **Example Pipeline** – `scripts/main.py` demonstrates how to search the graph
  with both strategies, generate answers and let the judge agent evaluate them.

## Requirements

- Python **3.11**
- [Neo4j](https://neo4j.com/) 5.x accessible via Bolt
- An API key for either **OpenAI** (`OPENAI_API_KEY`) or
  **OpenRouter** (`OPENROUTER_API_KEY`)

The project uses [`uv`](https://github.com/astral-sh/uv) for dependency
management. The required packages are declared in `pyproject.toml` and
`uv.lock`.

## Installation

1. Clone the repository and change into the project directory.
2. Install `uv` and synchronise the environment:

   ```bash
   pip install uv
   uv sync
   ```
3. Ensure a Neo4j instance is running. You can use the provided
   `docker-compose.yaml` which starts Neo4j and the application container:

   ```bash
   docker compose up -d
   ```

   The compose file exposes Neo4j on `localhost:7474` and `localhost:7687` with
   default credentials `neo4j/password`.

4. Set the required environment variables, for example:

   ```bash
   export OPENROUTER_API_KEY=<your-key>
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USERNAME=neo4j
   export NEO4J_PASSWORD=password
   ```

## Usage

1. **Generate the graph** (only needs to be run when the source text changes):

   ```bash
   python -m scripts.graph_generator
   ```

   This reads the files in `scripts/data`, creates embeddings and hierarchical
   summaries, and stores everything in Neo4j.

2. **Run the example pipeline** which queries the graph using different search
   strategies and compares the answers:

   ```bash
   python -m scripts.main
   ```

   The output shows the answers returned by each strategy as well as the
   judgement produced by the `JudgeAgent`.

## Project Structure

```
scripts/
├── agent/                 # QAAgent and JudgeAgent
├── data/                  # Example text files used to build the graph
├── graphdb/               # Neo4j client helper
├── model/                 # Embedding and language model utilities
├── plugins/               # Graph search strategies
├── graph_generator.py     # Creates the knowledge graph
└── main.py                # Example workflow
```

## License

This project is provided for educational purposes and does not include any
production guarantees.
