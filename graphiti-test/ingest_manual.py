"""
ingest_manual.py
Ingests RayJai's training manual into the knowledge graph by section.
Run: uv run python ingest_manual.py
"""
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# reuse same HD schema from ingest_transcript.py
from ingest_transcript import (
    HD_ENTITY_TYPES,
    HD_EDGE_TYPES,
    HD_EDGE_TYPE_MAP,
    HD_EXTRACTION_INSTRUCTIONS,
    embedder,
    cross_encoder,
)

load_dotenv()

MANUAL_FILE = "RayJai's reading training manual.docx.txt"

llm_client = AnthropicClient(
    config=LLMConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5",
        small_model="claude-haiku-4-5",
    )
)

graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=cross_encoder,
)

def chunk_manual(text: str) -> list[dict]:
    """Split manual into sections by the numbered headings."""
    import re
    # Split on section markers like "01  How to Use" or "4.1  The Body Graph"
    sections = re.split(r'\n(?=\d+[\.\s]+\s+[A-Z])', text)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) < 100:  # skip very short fragments
            continue
        # use first line as title
        title = section.split('\n')[0].strip()[:80]
        chunks.append({
            "name": f"manual-section-{i:03d}",
            "title": title,
            "text": section,
        })
    return chunks

async def ingest():
    path = Path(MANUAL_FILE)
    if not path.exists():
        print(f"Error: {MANUAL_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    chunks = chunk_manual(text)
    print(f"Split manual into {len(chunks)} sections")

    print("Building Neo4j indices and constraints...")
    await graphiti.build_indices_and_constraints()

    ref_time = datetime.now(timezone.utc)
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{total}] {chunk['title']}")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await graphiti.add_episode(
                    name=chunk["name"],
                    episode_body=chunk["text"],
                    source_description="The 26•44 AI Agent Training Manual — RayJai Babauta methodology",
                    reference_time=ref_time,
                    group_id="the2644-manual",
                    entity_types=HD_ENTITY_TYPES,
                    edge_types=HD_EDGE_TYPES,
                    edge_type_map=HD_EDGE_TYPE_MAP,
                    custom_extraction_instructions=HD_EXTRACTION_INSTRUCTIONS,
                )
                break
            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                wait = 15 * (2 ** attempt) if is_rate_limit else 15
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries - 1} in {wait}s — {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    raise
        await asyncio.sleep(15)

    await graphiti.close()
    print(f"\nDone! {total} sections ingested.")

if __name__ == "__main__":
    asyncio.run(ingest())