"""
ingest_transcript.py
Reads transcript_episodes.json (produced by parse_transcript.py) and feeds
each speaker turn into Graphiti as a timestamped episode.

Uses Ollama/gemma4:26b for LLM (entity extraction) and Ollama/nomic-embed-text for embeddings.
All inference is fully local — no external API calls required.

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

import openai
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

EPISODES_FILE = "transcript_episodes.json"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3:30b"

# ── LLM: Ollama/gemma4:26b (local) ────────────────────────────────────────────
llm_client = OpenAIGenericClient(config=LLMConfig(
    api_key="ollama",
    model=OLLAMA_MODEL,
    small_model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
))

# ── Embeddings: Ollama/nomic-embed-text (local) ───────────────────────────────
embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key="ollama",
    embedding_model="nomic-embed-text",
    embedding_dim=768,
    base_url=OLLAMA_BASE_URL,
))

# ── Cross-encoder: Ollama/gemma4:26b (local) ──────────────────────────────────
cross_encoder = OpenAIRerankerClient(
    client=llm_client,
    config=llm_client.config,
)

# ── Raw OpenAI-compatible client for the segment classifier ───────────────────
_ollama = openai.AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

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
    type_name: str = Field(description="The Human Design type name e.g. Manifesting Generator")
    strategy: Optional[str] = Field(default=None, description="The type's strategy e.g. Respond, Wait for Invitation")
    not_self_theme: Optional[str] = Field(default=None, description="The not-self theme e.g. Frustration, Bitterness")
    signature: Optional[str] = Field(default=None, description="The signature e.g. Satisfaction, Success")

class HDCenter(BaseModel):
    center_name: str = Field(description="Centre name e.g. Sacral, Solar Plexus, G-Center")
    defined_or_open: Optional[str] = Field(default=None, description="Whether the centre is defined or open/undefined")
    function: Optional[str] = Field(default=None, description="What this centre governs")

class HDGate(BaseModel):
    number: str = Field(description="Gate number e.g. 26, 44")
    gate_name: Optional[str] = Field(default=None, description="Gate name e.g. The Accumulator")
    center: Optional[str] = Field(default=None, description="Which centre this gate belongs to")
    gift: Optional[str] = Field(default=None, description="The gift expression of this gate")

class HDChannel(BaseModel):
    channel_name: Optional[str] = Field(default=None, description="Channel name")
    gate_1: str = Field(description="First gate number")
    gate_2: str = Field(description="Second gate number")
    theme: Optional[str] = Field(default=None, description="The theme or energy of this channel")

class HDProfile(BaseModel):
    lines: str = Field(description="Profile line combination e.g. 3/5, 6/2")
    conscious_line: Optional[str] = Field(default=None, description="The conscious personality line")
    unconscious_line: Optional[str] = Field(default=None, description="The unconscious design line")
    archetype: Optional[str] = Field(default=None, description="RayJai's reframe of this profile")

class HDAuthority(BaseModel):
    authority_name: str = Field(description="Authority type e.g. Emotional, Sacral, Splenic")
    decision_process: Optional[str] = Field(default=None, description="How this authority makes decisions")

class HDConcept(BaseModel):
    concept_name: str = Field(description="HD concept name e.g. deconditioning, not-self, openness")
    definition: Optional[str] = Field(default=None, description="What this concept means", max_length=500)
    rayjai_reframe: Optional[str] = Field(default=None, description="RayJai's specific language or reframe for this concept", max_length=500)

class RayJaiTeaching(BaseModel):
    insight: str = Field(description="The specific teaching, reframe, or metaphor RayJai used", max_length=500)
    context: Optional[str] = Field(default=None, description="What prompted this teaching")
    related_hd_concept: Optional[str] = Field(default=None, description="Which HD concept this relates to")

class Person(BaseModel):
    person_name: str = Field(description="Person's name")
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

# ── HD Edge Types ─────────────────────────────────────────────────────────────
class DefinedIn(BaseModel):
    """A gate is defined in a centre"""
    confidence: Optional[float] = Field(default=None, description="Confidence score 0-1")

class ConnectsTo(BaseModel):
    """A gate connects to another gate forming a channel"""
    channel_name: Optional[str] = Field(default=None, description="The resulting channel name")

class AuthorityFor(BaseModel):
    """An authority type belongs to a HD type"""
    confidence: Optional[float] = Field(default=None)

class StrategyFor(BaseModel):
    """A strategy belongs to a HD type"""
    confidence: Optional[float] = Field(default=None)

class NotSelfOf(BaseModel):
    """A not-self theme belongs to a HD type"""
    confidence: Optional[float] = Field(default=None)

class SignatureOf(BaseModel):
    """A signature belongs to a HD type"""
    confidence: Optional[float] = Field(default=None)

class TeachesThrough(BaseModel):
    """A RayJai teaching is delivered through a HD concept"""
    confidence: Optional[float] = Field(default=None)

class Reframes(BaseModel):
    """A RayJai teaching reframes a HD concept with his proprietary language"""
    original_term: Optional[str] = Field(default=None, description="The original HD term being reframed")

class Illustrates(BaseModel):
    """A RayJai teaching illustrates a centre, type, gate or concept"""
    confidence: Optional[float] = Field(default=None)

class BuildsOn(BaseModel):
    """A concept builds on another concept — must understand the second to grasp the first"""
    confidence: Optional[float] = Field(default=None)

class GrantsPermission(BaseModel):
    """A RayJai teaching grants permission to a person or type"""
    permission_statement: Optional[str] = Field(default=None, description="What permission is being granted")

class ResonatedWith(BaseModel):
    """A teaching or concept resonated with a person in session"""
    confidence: Optional[float] = Field(default=None)

class ConditioningOf(BaseModel):
    """An open centre is a source of conditioning for a person"""
    confidence: Optional[float] = Field(default=None)

class RelatesTo(BaseModel):
    """Generic fallback relationship between two entities"""
    confidence: Optional[float] = Field(default=None)

HD_EDGE_TYPES = {
    "DefinedIn": DefinedIn,
    "ConnectsTo": ConnectsTo,
    "AuthorityFor": AuthorityFor,
    "StrategyFor": StrategyFor,
    "NotSelfOf": NotSelfOf,
    "SignatureOf": SignatureOf,
    "TeachesThrough": TeachesThrough,
    "Reframes": Reframes,
    "Illustrates": Illustrates,
    "BuildsOn": BuildsOn,
    "GrantsPermission": GrantsPermission,
    "ResonatedWith": ResonatedWith,
    "ConditioningOf": ConditioningOf,
    "RelatesTo": RelatesTo,
}

HD_EDGE_TYPE_MAP = {
    ("HDGate", "HDCenter"): ["DefinedIn"],
    ("HDGate", "HDGate"): ["ConnectsTo"],
    ("HDAuthority", "HDType"): ["AuthorityFor"],
    ("HDType", "HDType"): ["StrategyFor", "NotSelfOf", "SignatureOf"],
    ("HDConcept", "HDType"): ["NotSelfOf", "SignatureOf", "BuildsOn"],
    ("RayJaiTeaching", "HDConcept"): ["TeachesThrough", "Reframes", "Illustrates"],
    ("RayJaiTeaching", "HDCenter"): ["Illustrates"],
    ("RayJaiTeaching", "HDType"): ["Illustrates", "GrantsPermission"],
    ("RayJaiTeaching", "HDGate"): ["Illustrates"],
    ("RayJaiTeaching", "Person"): ["GrantsPermission"],
    ("HDConcept", "HDConcept"): ["BuildsOn"],
    ("HDCenter", "Person"): ["ConditioningOf"],
    ("RayJaiTeaching", "HDProfile"): ["Illustrates"],
    ("HDProfile", "HDType"): ["BuildsOn"],
    ("Person", "HDConcept"): ["ResonatedWith"],
    ("Person", "RayJaiTeaching"): ["ResonatedWith"],
    ("Entity", "Entity"): ["RelatesTo"],
}

HD_EXTRACTION_INSTRUCTIONS = """
You are extracting structured entities from a Human Design reading session transcript.

CRITICAL: You MUST populate the specific attributes for each entity type. Do not leave attributes empty.
Pull attribute values DIRECTLY from the surrounding prose — use RayJai's own words where possible.
Definitions and insights should be full sentences, not just labels.

For each entity found, extract:
- HDType: populate type_name, strategy, not_self_theme, signature
- HDCenter: populate center_name, defined_or_open, function (describe what the centre governs in 1-2 sentences)
- HDGate: populate number, gate_name, center, gift
- HDChannel: populate gate_1, gate_2, channel_name, theme
- HDProfile: populate lines, conscious_line, unconscious_line, archetype
- HDAuthority: populate authority_name, decision_process
- HDConcept: populate concept_name, definition (full sentence from the text), rayjai_reframe (RayJai's exact words)
- RayJaiTeaching: populate insight with RayJai's EXACT teaching, reframe or metaphor — this is the most important field
- Person: populate person_name, role (client or practitioner)

CONCRETE EXAMPLES of well-populated nodes:

HDType:
  type_name: "Manifesting Generator"
  strategy: "Respond to life — wait for something to light you up, then act"
  not_self_theme: "Frustration when initiating without waiting to respond"
  signature: "Satisfaction and bursts of sustainable energy"

HDCenter:
  center_name: "Solar Plexus"
  defined_or_open: "defined"
  function: "The emotional authority centre — it rides a wave of highs and lows, and wisdom only comes when the wave has moved through, never in the moment of emotional charge"

HDCenter (open):
  center_name: "Sacral"
  defined_or_open: "open"
  function: "An undefined Sacral amplifies and reflects the life-force energy of defined Sacrals around it — it is not a reliable source of sustainable work energy for this person"

HDGate:
  number: "26"
  gate_name: "The Accumulator"
  center: "Heart"
  gift: "The ability to convince others and sell ideas with integrity when acting from the heart"

HDConcept:
  concept_name: "deconditioning"
  definition: "The multi-year process of shedding the conditioning absorbed from open centres and living as others, returning to your authentic design"
  rayjai_reframe: "Peeling back the layers of who you were told to be so you can land in who you actually are"

HDConcept:
  concept_name: "not-self"
  definition: "The voice or behaviour pattern that runs when a person is living out of alignment with their design — driven by conditioning from open centres"
  rayjai_reframe: "The imposter running the show — the you that learned to survive, not the you that was born to thrive"

RayJaiTeaching:
  insight: "Your open Head Centre is not a problem — it means you are a phenomenal receiver of inspiration. The key is: inspiration visits you, it doesn't live here. You don't need to solve every question that walks in."
  context: "Client expressing anxiety about constant mental chatter"
  related_hd_concept: "open Head Centre conditioning"

HDAuthority:
  authority_name: "Emotional"
  decision_process: "Ride the emotional wave fully before deciding — clarity comes at the bottom of the wave, not in the high or the low. Sleep on it. If it still feels right after the wave has passed, it is correct."

HDProfile:
  lines: "3/5"
  conscious_line: "Line 3 — the Martyr, learning through trial and error and lived experience"
  unconscious_line: "Line 5 — the Heretic, projected onto by others as a practical problem-solver"
  archetype: "The Resilient Revolutionary — someone whose mess becomes their medicine"

Person:
  person_name: "Linzie Lee"
  role: "client"
  hd_type: "Manifesting Generator"

Focus especially on RayJaiTeaching nodes — capture RayJai's specific language, metaphors and reframes verbatim.
Do NOT use meta-language like "no new attributes" or "entity unchanged" — only extract real content from the transcript.

RELATIONSHIP EXTRACTION — CRITICAL:
You must also identify relationships between entities. Use the relationship type names EXACTLY as listed below.

Available relationship types and when to use them:
- DefinedIn       — a Gate belongs to a Centre (e.g. Gate 26 is in the Heart Centre)
- ConnectsTo      — a Gate connects to another Gate forming a Channel
- AuthorityFor    — an Authority type belongs to an HD Type (e.g. Emotional authority → Generator)
- StrategyFor     — a strategy concept links to an HD Type
- NotSelfOf       — a not-self theme or concept belongs to an HD Type
- SignatureOf     — a signature belongs to an HD Type
- TeachesThrough  — a RayJai teaching is delivered through a HD concept
- Reframes        — a RayJai teaching reframes a HD concept in his own language
- Illustrates     — a RayJai teaching illustrates a centre, type, gate, profile or concept
- BuildsOn        — a concept builds on another concept (must understand B to grasp A)
- GrantsPermission — a RayJai teaching grants explicit permission to a person or type
- ResonatedWith   — a teaching or concept resonated with the client in session
- ConditioningOf  — an open centre is a conditioning source for a person
- RelatesTo       — generic fallback; use ONLY when no other type fits

CONCRETE EXAMPLES of correct relationship extraction:

SOURCE TEXT: "Gate 26 which is in the heart centre — this is the gate of the Accumulator"
→ HDGate(number="26") -[DefinedIn]-> HDCenter(center_name="Heart")

SOURCE TEXT: "Your Sacral is undefined, which means you absorb and amplify others' energy"
→ HDCenter(center_name="Sacral", defined_or_open="open") -[ConditioningOf]-> Person(person_name="client")

SOURCE TEXT: "As an Emotional authority you ride a wave — never decide in the high or the low"
→ HDAuthority(authority_name="Emotional") -[AuthorityFor]-> HDType(type_name="Generator")
→ RayJaiTeaching(insight="Never decide in the high or the low...") -[Illustrates]-> HDAuthority(authority_name="Emotional")

SOURCE TEXT: "RayJai says: your open Head Centre is not broken — it means inspiration visits you, it doesn't live here"
→ RayJaiTeaching(insight="Your open Head Centre is not broken — inspiration visits you, it doesn't live here") -[Reframes]-> HDConcept(concept_name="open Head Centre")
→ RayJaiTeaching(insight="...") -[GrantsPermission]-> Person(person_name="client")

SOURCE TEXT: "The 3/5 profile means you learn through trial and error — your mess becomes your medicine"
→ RayJaiTeaching(insight="Your mess becomes your medicine") -[Illustrates]-> HDProfile(lines="3/5")
→ HDProfile(lines="3/5") -[BuildsOn]-> HDType(type_name="Manifesting Generator")

SOURCE TEXT: "Gate 26 and Gate 44 together form the channel of Surrender"
→ HDGate(number="26") -[ConnectsTo]-> HDGate(number="44")

When in doubt between two types, prefer the more specific one over RelatesTo.
"""


# ── Episode windowing ─────────────────────────────────────────────────────────
def group_episodes(episodes: list[dict], window_size: int = 5) -> list[dict]:
    """Group consecutive speaker turns into windows for richer extraction context.

    A single isolated turn like 'wow,' or 'yeah, okay' gives the LLM nothing to
    extract. Grouping 5 turns together means each episode contains enough
    conversational context for entities and relationships to be identified.
    Also reduces total API calls ~5x, keeping token usage within rate limits.
    """
    segments = []
    for i in range(0, len(episodes), window_size):
        window = episodes[i : i + window_size]
        body = "\n".join(f"{ep['speaker']}: {ep['text']}" for ep in window)
        segments.append({
            "name": f"segment-{i // window_size:04d}",
            "body": body,
            "timestamp": window[0]["timestamp"],
        })
    return segments


# ── Turn classifier ───────────────────────────────────────────────────────────
async def classify_segment(body: str) -> tuple[str, str]:
    """Return (decision, reason) where decision is RELEVANT or SKIP.

    RELEVANT — segment contains any Human Design concepts, teachings, chart
               readings, or meaningful client reflections worth ingesting.
    SKIP     — the ENTIRE segment is small talk, greetings, tech checks,
               scheduling, or off-topic conversation with zero HD content.
    """
    response = await _ollama.chat.completions.create(
        model=OLLAMA_MODEL,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": (
                f"{body}\n\n"
                "Classify this transcript segment. Reply with EXACTLY one of:\n"
                "  RELEVANT\n"
                "  SKIP: <brief reason>\n"
                "Mark RELEVANT if ANY part contains Human Design concepts, teachings, "
                "chart readings, or meaningful reflections on HD or personal growth. "
                "Mark SKIP only if the ENTIRE segment is small talk, greetings, "
                "tech/audio checks, scheduling, or purely off-topic chatter.\n"
                "Nothing else. No explanation outside that format."
            ),
        }],
    )
    raw = response.choices[0].message.content.strip()
    upper = raw.upper()
    if "RELEVANT" in upper:
        return "RELEVANT", ""
    if "SKIP" in upper:
        parts = raw.split(":", 1)
        reason = parts[1].strip() if len(parts) > 1 else ""
        return "SKIP", reason
    # unrecognised response — default to RELEVANT so we don't lose data
    return "RELEVANT", ""


# ── Ingestion ─────────────────────────────────────────────────────────────────
async def ingest(episodes: list[dict]):
    print("Building Neo4j indices and constraints...")
    await graphiti.build_indices_and_constraints()

    segments = group_episodes(episodes, window_size=5)
    total = len(segments)
    skipped = 0
    print(f"Grouped {len(episodes)} turns into {total} segments of ~5 turns each.\n")

    for i, seg in enumerate(segments):
        episode_name = seg["name"]
        body = seg["body"]
        ref_time = datetime.strptime(seg["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

        print(f"[{i+1}/{total}] {episode_name}")

        decision, reason = await classify_segment(body)
        if decision == "SKIP":
            skipped += 1
            print(f"  [SKIP] {episode_name} — {reason}")
            continue

        max_retries = 5
        for attempt in range(max_retries):
            try:
                await graphiti.add_episode(
                    name=episode_name,
                    episode_body=body,
                    source_description="The 26•44 session transcript — RayJai Babauta Human Design reading",
                    reference_time=ref_time,
                    group_id="the2644-sessions",
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
        await asyncio.sleep(2)

    await graphiti.close()
    ingested = total - skipped
    print(f"\nDone! {ingested}/{total} segments ingested ({skipped} skipped).")
    print("  http://localhost:7474")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m")


def main():
   
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