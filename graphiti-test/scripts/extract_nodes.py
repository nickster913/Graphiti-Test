"""
extract_nodes.py

Reads output/transcript_chunks.json (produced by chunk_transcript.py), sends
each chunk to the Claude API for HD entity / relationship extraction, then
merges all results into output/hd_graph_seed.json.

Usage:
  python scripts/extract_nodes.py

Environment:
  ANTHROPIC_API_KEY — required

Output:
  output/hd_graph_seed.json (merged with any existing content)
"""

import json
import os
import re
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_FILE = _PROJECT_ROOT / "output" / "transcript_chunks.json"
OUTPUT_FILE = _PROJECT_ROOT / "output" / "hd_graph_seed.json"
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 1  # seconds

SYSTEM_PROMPT = """\
You are a knowledge graph extraction specialist for Human Design (HD).
Extract entities and relationships from the transcript chunk provided.
Return ONLY valid JSON. No preamble, no explanation, no markdown code blocks.
The JSON must follow this exact schema:
{
  "nodes": [...],
  "edges": [...]
}"""

USER_PROMPT_TEMPLATE = """\
Extract all HD entities and relationships from this transcript chunk.

NODE TYPES AND THEIR FIELDS:
- Person: name, role (practitioner/client), hd_type
- HDType: name, type_name, strategy, not_self_theme, signature
- HDCenter: name, center_name, defined_or_open, function
- HDGate: name, number, gate_name, center, gift
- HDChannel: name, gate_1, gate_2, theme
- HDProfile: name, lines, conscious_line, unconscious_line, archetype
- HDAuthority: name, authority_name, decision_process
- HDConcept: name, definition, rayjai_reframe
- RayJaiTeaching: name (short label), insight (RayJai's exact words), context, related_hd_concept

EDGE TYPES (from → to):
- DEFINED_IN: HDGate → HDCenter
- CONNECTS_TO: HDGate → HDGate
- FORMS: HDGate → HDChannel
- DEFINES: HDCenter → HDType
- TYPE_OF: HDType → Person
- BELONGS_TO: HDProfile → Person
- ACTIVE_IN: HDGate → Person
- CONDITIONING_OF: HDCenter → Person
- ILLUSTRATES: RayJaiTeaching → any HD node
- REFRAMES: RayJaiTeaching → HDConcept
- GRANTS_PERMISSION: RayJaiTeaching → Person or HDType
- TEACHES_THROUGH: RayJaiTeaching → HDConcept
- PART_OF: HDGate or HDCenter → HDConcept
- BUILDS_ON: HDConcept → HDConcept

RULES:
- Every node must have a "type" field matching one of the node types above
- Every node must have a "name" field — this is the unique identifier
- Every edge must have "from", "to", and "type" fields
- "from" and "to" must exactly match node "name" values
- Capture RayJai's EXACT words in RayJaiTeaching insight fields
- Only extract what is explicitly in the transcript — do not invent
- If nothing HD-related is in the chunk, return {{"nodes": [], "edges": []}}

TRANSCRIPT CHUNK:
{chunk_body}"""

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

# Strips markdown code fences if the model wraps in them despite instructions.
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(raw: str) -> dict:
    """
    Parse JSON from a raw API response string.
    Handles accidental markdown fences.
    Raises ValueError if no valid JSON object is found.
    """
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)

# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_node_fields(existing: dict, incoming: dict) -> dict:
    """
    Merge two nodes with the same name.
    Keep non-null values from either side; prefer the more detailed (longer) string.
    """
    merged = dict(existing)
    for key, val in incoming.items():
        if key not in merged or merged[key] is None:
            merged[key] = val
        elif val is not None and isinstance(val, str) and isinstance(merged[key], str):
            # Keep the longer / more detailed string.
            if len(val) > len(merged[key]):
                merged[key] = val
    return merged


def merge_graphs(base: dict, incoming: dict) -> dict:
    """
    Merge incoming nodes/edges into base, deduplicating as described.
    """
    # -- Nodes --
    node_map: dict[str, dict] = {n["name"]: n for n in base.get("nodes", [])}
    for node in incoming.get("nodes", []):
        name = node.get("name")
        if not name:
            continue
        if name in node_map:
            node_map[name] = merge_node_fields(node_map[name], node)
        else:
            node_map[name] = node

    # -- Edges --
    edge_keys: set[tuple] = {
        (e["from"], e["type"], e["to"]) for e in base.get("edges", [])
    }
    merged_edges: list[dict] = list(base.get("edges", []))
    for edge in incoming.get("edges", []):
        key = (edge.get("from"), edge.get("type"), edge.get("to"))
        if None in key:
            continue
        if key not in edge_keys:
            edge_keys.add(key)
            merged_edges.append(edge)

    return {"nodes": list(node_map.values()), "edges": merged_edges}

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_claude(client: anthropic.Anthropic, chunk_body: str) -> dict:
    """
    Send one chunk to Claude and return the parsed {nodes, edges} dict.
    Raises ValueError if the response cannot be parsed as JSON.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(chunk_body=chunk_body)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = message.content[0].text
    return extract_json(raw)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    chunks_path = Path(CHUNKS_FILE)
    if not chunks_path.exists():
        raise SystemExit(
            f"Error: {CHUNKS_FILE} not found. Run chunk_transcript.py first."
        )

    with chunks_path.open(encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    # Load existing graph (if present) as the merge base.
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            graph = json.load(f)
        print(f"Loaded existing {OUTPUT_FILE}: "
              f"{len(graph.get('nodes', []))} nodes, "
              f"{len(graph.get('edges', []))} edges")
    else:
        graph = {"nodes": [], "edges": []}
        print(f"No existing {OUTPUT_FILE} found — starting fresh.")

    client = anthropic.Anthropic(api_key=api_key)
    total = len(chunks)

    nodes_before = len(graph["nodes"])
    edges_before = len(graph["edges"])
    skipped = 0

    for i, chunk in enumerate(chunks):
        chunk_index = chunk["chunk_index"]
        char_count = chunk["char_count"]
        print(f"[{i + 1}/{total}] Extracting chunk {chunk_index} ({char_count:,} chars)...",
              end=" ", flush=True)

        try:
            result = call_claude(client, chunk["body"])
        except Exception as exc:
            print(f"ERROR — {exc}")
            skipped += 1
            continue

        n_nodes = len(result.get("nodes", []))
        n_edges = len(result.get("edges", []))
        print(f"→ {n_nodes} nodes, {n_edges} edges")

        graph = merge_graphs(graph, result)

        if i < total - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Write output.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    nodes_after = len(graph["nodes"])
    edges_after = len(graph["edges"])
    nodes_added = nodes_after - nodes_before
    edges_added = edges_after - edges_before

    print()
    print("=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Chunks processed : {total - skipped}/{total}  ({skipped} skipped)")
    print(f"Total nodes      : {nodes_after}  (+{nodes_added} new)")
    print(f"Total edges      : {edges_after}  (+{edges_added} new)")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
