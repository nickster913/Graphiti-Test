# graphiti-test

A local experiment for building a knowledge graph using [Graphiti](https://github.com/getzep/graphiti) with fully local models via [Ollama](https://ollama.com/) and [Neo4j](https://neo4j.com/).

## Overview

This project demonstrates how to use `graphiti-core` to ingest unstructured text episodes into a temporal knowledge graph, with all LLM inference and embeddings running locally — no OpenAI API key required.

- **LLM**: `gemma4:26b` served via Ollama (OpenAI-compatible endpoint)
- **Embeddings**: `nomic-embed-text` via Ollama (768-dimensional vectors)
- **Graph database**: Neo4j (bolt on `localhost:7687`)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com/) running locally with the required models pulled
- [Neo4j](https://neo4j.com/download/) running locally

### Pull required Ollama models

```bash
ollama pull gemma4:26b
ollama pull nomic-embed-text
```

### Start Neo4j

Ensure Neo4j is running and accessible at `bolt://localhost:7687` with the default credentials (`neo4j` / `password`), or update `test.py` to match your setup.

## Installation

```bash
uv sync
```

## Usage

Run the test script to ingest a sample episode into the knowledge graph:

```bash
uv run python test.py
```

This will:
1. Build Neo4j indices and constraints.
2. Add a sample episode describing two people (Alice and Bob) and their relationship.
3. Print a confirmation message when complete.

Once finished, open the [Neo4j Browser](http://localhost:7474) to explore the resulting knowledge graph.

## Project Structure

```
graphiti-test/
├── test.py          # Main experiment script
├── main.py          # Project entry point placeholder
├── pyproject.toml   # Project metadata and dependencies
└── uv.lock          # Locked dependency versions
```

## Dependencies

| Package | Purpose |
|---|---|
| `graphiti-core` | Knowledge graph construction and querying |
| `python-dotenv` | Environment variable management |
