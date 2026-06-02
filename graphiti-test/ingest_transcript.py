"""
ingest_transcript.py
Reads transcript_episodes.json (produced by parse_transcript.py) and feeds
each speaker turn into Graphiti as a timestamped episode.

Uses Claude API for LLM (entity extraction) and Ollama/nomic-embed-text for embeddings.

Run:
    uv run python parse_transcript.py
    uv run python ingest_transcript.py
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

EPISODES_FILE = "transcript_episodes.json"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── LLM: Claude API for entity extraction ────────────────────────────────────
llm_client = AnthropicClient(
    config=LLMConfig(
        api_key=ANTHROPIC_API_KEY,
        model="claude-sonnet-4-6",
        small_model="claude-haiku-4-5",
    )
)

# ── Embeddings: Ollama/nomic-embed-text (local) ───────────────────────────────
embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key="ollama",
    embedding_model="nomic-embed-text",
    embedding_dim=768,
    base_url="http://localhost:11434/v1",
))

# ── Cross-encoder: reuse Claude via OpenAI-compatible wrapper ─────────────────
ollama_llm_config = LLMConfig(
    api_key="ollama",
    model="gemma4:26b",
    small_model="gemma4:26b",
    base_url="http://localhost:11434/v1",
)
ollama_client = OpenAIGenericClient(config=ollama_llm_config)
cross_encoder = OpenAIRerankerClient(
    client=ollama_client,
    config=ollama_llm_config,
)

# ── Graphiti setup ────────────────────────────────────────────────────────────
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=cross_encoder,
)

# ── HD Entity Schema ──────────────────────────────────────────────────────────
class HDType(BaseModel):
    name: str = Field(description="The Human Design type name e.g. Manifesting Generator")
    strategy: Optional[str] = Field(default=None, description="The type's strategy e.g. Respond, Wait for Invitation")
    not_self_theme: Optional[str] = Field(default=None, description="The not-self theme e.g. Frustration, Bitterness")
    signature: Optional[str] = Field(default=None, description="The signature e.g. Satisfaction, Success")

class HDCenter(BaseModel):
    name: str = Field(description="Centre name e.g. Sacral, Solar Plexus, G-Center")
    defined_or_open: Optional[str] = Field(default=None, description="Whether the centre is defined or open/undefined")
    function: Optional[str] = Field(default=None, description="What this centre governs")

class HDGate(BaseModel):
    number: str = Field(description="Gate number e.g. 26, 44")
    name: Optional[str] = Field(default=None, description="Gate name e.g. The Accumulator")
    center: Optional[str] = Field(default=None, description="Which centre this gate belongs to")
    gift: Optional[str] = Field(default=None, description="The gift expression of this gate")

class HDChannel(BaseModel):
    name: Optional[str] = Field(default=None, description="Channel name")
    gate_1: str = Field(description="First gate number")
    gate_2: str = Field(description="Second gate number")
    theme: Optional[str] = Field(default=None, description="The theme or energy of this channel")

class HDProfile(BaseModel):
    lines: str = Field(description="Profile line combination e.g. 3/5, 6/2")
    conscious_line: Optional[str] = Field(default=None, description="The conscious personality line")
    unconscious_line: Optional[str] = Field(default=None, description="The unconscious design line")
    archetype: Optional[str] = Field(default=None, description="RayJai's reframe of this profile")

class HDAuthority(BaseModel):
    name: str = Field(description="Authority type e.g. Emotional, Sacral, Splenic")
    decision_process: Optional[str] = Field(default=None, description="How this authority makes decisions")

class HDConcept(BaseModel):
    name: str = Field(description="HD concept name e.g. deconditioning, not-self, openness")
    definition: Optional[str] = Field(default=None, description="What this concept means")
    rayjai_reframe: Optional[str] = Field(default=None, description="RayJai's specific language or reframe for this concept")

class RayJaiTeaching(BaseModel):
    insight: str = Field(description="The specific teaching, reframe, or metaphor RayJai used")
    context: Optional[str] = Field(default=None, description="What prompted this teaching")
    related_hd_concept: Optional[str] = Field(default=None, description="Which HD concept this relates to")

class Person(BaseModel):
    name: str = Field(description="Person's name")
    role: Optional[str] = Field(default=None, description="Role e.g. client, practitioner")
    hd_type: Optional[str] = Field(default=None, description="Their Human Design type if mentioned")

HD_ENTITY_TYPES = {
    "HDType": HDType,
    "HDCenter": HDCenter,
    "HDGate": HDGate,
    "HDChannel": HDChannel,
    "HDProfile": HDProfile,
    "HDAuthority": HDAuthority,
    "HDConcept": HDConcept,
    "RayJaiTeaching": RayJaiTeaching,
    "Person": Person,
}

HD_EXTRACTION_INSTRUCTIONS = """
You are extracting entities from a Human Design reading session transcript.
Focus on:
- HD types, centres, gates, channels, profiles, and authorities mentioned
- RayJai's specific reframes and teaching moments (these are valuable IP)
- How concepts are explained in plain language, not textbook HD
- Relationships between concepts (e.g. which gates form which channels, which authority belongs to which type)
- Client responses that indicate resonance or confusion with a concept
Do NOT extract generic entities like locations or organisations unless directly relevant to HD.
"""


# ── Ingestion ─────────────────────────────────────────────────────────────────
async def ingest(episodes: list[dict]):
    print("Building Neo4j indices and constraints...")
    await graphiti.build_indices_and_constraints()

    total = len(episodes)
    for i, ep in enumerate(episodes):
        episode_name = f"turn-{i:04d}"
        body = f"{ep['speaker']}: {ep['text']}"
        ref_time = datetime.strptime(ep["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

        print(f"[{i+1}/{total}] {episode_name} — {ep['speaker']} ({ep['timestamp']})")
        await graphiti.add_episode(
            name=episode_name,
            episode_body=body,
            source_description="The 26•44 session transcript — RayJai Babauta Human Design reading",
            reference_time=ref_time,
            group_id="the2644-sessions",
            entity_types=HD_ENTITY_TYPES,
            custom_extraction_instructions=HD_EXTRACTION_INSTRUCTIONS,
        )

    await graphiti.close()
    print(f"\nDone! {total} episodes ingested. Open Neo4j Browser to explore the graph:")
    print("  http://localhost:7474")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m")


def main():
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    path = Path(EPISODES_FILE)
    if not path.exists():
        print(f"Error: {EPISODES_FILE} not found.", file=sys.stderr)
        print("Run `uv run python parse_transcript.py` first.", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        episodes = json.load(f)

    if not episodes:
        print("No episodes found in JSON file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(episodes)} episodes from {EPISODES_FILE}")
    asyncio.run(ingest(episodes))


if __name__ == "__main__":
    main()