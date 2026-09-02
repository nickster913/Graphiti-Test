"""
embed_graph.py

Adds a semantic-retrieval (RAG) layer on top of the Neo4j knowledge graph.

For each relevant node it computes an embedding of the node's verbatim text with
a local Ollama model (`nomic-embed-text`, 768-dim) and stores it on the node,
then creates Neo4j vector indexes so reader.py can do fast cosine similarity
search.

What gets embedded:
  - RayJaiTeaching.insight  -> RayJaiTeaching.embedding   (content + verbatim voice)
  - HDVoicePattern.phrase   -> HDVoicePattern.embedding   (voice exemplars)

Vector indexes created (cosine, 768-dim):
  - rayjai_teaching_embedding
  - hdvoice_embedding

Usage:
  uv run python scripts/embed_graph.py            # embed only nodes missing an embedding
  uv run python scripts/embed_graph.py --reembed  # recompute embeddings for every node

Requires:
  - Neo4j running (bolt://localhost:7687) with the graph already loaded
    (run load_graph.py first)
  - Ollama running with nomic-embed-text pulled
"""

from __future__ import annotations

import sys

from neo4j import GraphDatabase

from embed_utils import EMBED_DIM, embed_text

# ── Config ──────────────────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

TEACHING_INDEX = "rayjai_teaching_embedding"
VOICE_INDEX = "hdvoice_embedding"

# (label, text_property, index_name) for each embeddable node kind.
EMBED_TARGETS = [
    ("RayJaiTeaching", "insight", TEACHING_INDEX),
    ("HDVoicePattern", "phrase", VOICE_INDEX),
]


def create_vector_index(tx, index_name: str, label: str) -> None:
    tx.run(
        f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label}) ON (n.embedding)
        OPTIONS {{ indexConfig: {{
            `vector.dimensions`: {EMBED_DIM},
            `vector.similarity_function`: 'cosine'
        }} }}
        """
    )


def fetch_nodes_to_embed(tx, label: str, text_prop: str, reembed: bool) -> list[dict]:
    query = f"""
    MATCH (n:{label})
    WHERE n.{text_prop} IS NOT NULL AND n.{text_prop} <> ''
      AND ($reembed OR n.embedding IS NULL)
    RETURN elementId(n) AS id, n.{text_prop} AS text, n.name AS name
    """
    return tx.run(query, reembed=reembed).data()


def set_embedding(tx, node_id: str, vector: list[float]) -> None:
    tx.run(
        """
        MATCH (n) WHERE elementId(n) = $id
        CALL db.create.setNodeVectorProperty(n, 'embedding', $vector)
        """,
        id=node_id,
        vector=vector,
    )


def embed_label(driver, label: str, text_prop: str, index_name: str, reembed: bool) -> int:
    print(f"\n── {label} (embedding .{text_prop}) " + "─" * 20)

    with driver.session() as session:
        session.execute_write(create_vector_index, index_name, label)
        print(f"  [INDEX] ensured vector index {index_name!r}")

        rows = session.execute_read(fetch_nodes_to_embed, label, text_prop, reembed)

    if not rows:
        print("  Nothing to embed (all nodes already have embeddings).")
        return 0

    print(f"  Embedding {len(rows)} node(s)...")
    done = 0
    with driver.session() as session:
        for i, row in enumerate(rows, 1):
            try:
                vector = embed_text(row["text"])
            except Exception as exc:  # noqa: BLE001 - surface and continue
                print(f"  [ERROR] {row.get('name')!r}: {exc}")
                continue
            session.execute_write(set_embedding, row["id"], vector)
            done += 1
            if i % 25 == 0 or i == len(rows):
                print(f"    {i}/{len(rows)}")
    print(f"  Done: {done}/{len(rows)} embedded.")
    return done


def main() -> None:
    reembed = "--reembed" in sys.argv

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    total = 0
    try:
        for label, text_prop, index_name in EMBED_TARGETS:
            total += embed_label(driver, label, text_prop, index_name, reembed)
    finally:
        driver.close()

    print("\n" + "=" * 50)
    print("EMBEDDING COMPLETE")
    print("=" * 50)
    print(f"Nodes embedded : {total}")
    print(f"Vector indexes : {TEACHING_INDEX}, {VOICE_INDEX}")
    print("reader.py will now use semantic (vector) retrieval automatically.")


if __name__ == "__main__":
    main()
