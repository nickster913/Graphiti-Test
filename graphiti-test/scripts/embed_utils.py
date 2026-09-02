"""
embed_utils.py

Tiny shared helper for generating text embeddings with a local Ollama model
(`nomic-embed-text`, 768-dimensional). Used by embed_graph.py (to embed graph
nodes) and reader.py (to embed the user's question for vector search).

No extra Python package is required — this calls Ollama's HTTP API directly via
httpx (already installed transitively). You only need the Ollama server running
with the embedding model pulled:

    ollama pull nomic-embed-text

Environment (optional overrides):
  OLLAMA_HOST         — default "http://localhost:11434"
  OLLAMA_EMBED_MODEL  — default "nomic-embed-text"
"""

from __future__ import annotations

import os

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# nomic-embed-text produces 768-dimensional vectors. Keep this in sync with the
# `vector.dimensions` used when creating the Neo4j vector indexes.
EMBED_DIM = 768

_TIMEOUT = httpx.Timeout(60.0)


def embed_text(text: str) -> list[float]:
    """
    Return the embedding vector for `text`.

    Raises httpx.HTTPError / RuntimeError if Ollama is unreachable or returns
    an unexpected payload — callers that want graceful degradation should catch
    these (see reader.py's safe_embed).
    """
    text = (text or "").strip()
    if not text:
        return [0.0] * EMBED_DIM

    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    vector = resp.json().get("embedding")
    if not vector:
        raise RuntimeError(
            f"Ollama returned no embedding for model {EMBED_MODEL!r}. "
            "Is the model pulled? Try: ollama pull nomic-embed-text"
        )
    return vector
