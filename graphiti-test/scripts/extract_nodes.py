"""
extract_nodes.py

Reads output/transcript_chunks.json (produced by chunk_transcript.py), sends
each chunk to a local Ollama model for HD entity / relationship extraction, then
merges all results into output/hd_graph_seed.json.

Usage:
  python scripts/extract_nodes.py

Environment:
  OLLAMA_HOST           — optional, default http://localhost:11434
  OLLAMA_EXTRACT_MODEL  — optional, default qwen3.6:27b (see scripts/ollama_chat.py)

Output:
  output/hd_graph_seed.json (merged with any existing content)
"""

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from ollama_chat import EXTRACT_MODEL, chat_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_FILE = _PROJECT_ROOT / "output" / "transcript_chunks.json"
OUTPUT_FILE = _PROJECT_ROOT / "output" / "hd_graph_seed.json"
MODEL = EXTRACT_MODEL
SLEEP_BETWEEN_CALLS = 0  # seconds (local model — no rate limit)

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
- RayJaiTeaching: name (short label), insight (RayJai's exact words), context, related_hd_concept, trigger_context (emotional/situational trigger), effect (what the phrasing achieves: normalise/grant permission/reframe shame/etc)
- HDVoicePattern: name (short label), phrase (the exact phrase or pattern), usage_context (when to use it), trigger (what situation triggers it), avoid_if (when NOT to use it)
- HDSessionFlow: name (short label), step_number (integer), instruction (what to do), purpose (why), example_variant (an acceptable variation)
- HDBehaviourRule: name (short label), rule_type ("DO" or "DON'T"), rule (the rule itself), context (when it applies), rationale (why this rule exists)
- HDToneProfile: name (short label), emotional_state (what state the user is in), instruction (how to respond), example_response (an example of this tone in action)

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
- EXPRESSES: HDVoicePattern → HDType or HDConcept
- STEP_OF: HDSessionFlow → HDSessionFlow (ordering)
- GOVERNS: HDBehaviourRule → HDType or HDConcept or HDToneProfile
- CALIBRATES_FOR: HDToneProfile → HDType or Person

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
# Fix 1 — Auto-create stub nodes for dangling edge endpoints
# ---------------------------------------------------------------------------

def ensure_edge_nodes(graph: dict) -> dict:
    """
    For every node name referenced in an edge that has no corresponding node,
    auto-create a minimal HDConcept stub so that load_graph.py never fires a
    WARN for a missing node and no edge is silently dropped.
    """
    node_names: set[str] = {n["name"] for n in graph.get("nodes", [])}
    new_nodes: list[dict] = list(graph.get("nodes", []))
    stubs: list[str] = []

    for edge in graph.get("edges", []):
        for endpoint in (edge.get("from"), edge.get("to")):
            if endpoint and endpoint not in node_names:
                new_nodes.append({"type": "HDConcept", "name": endpoint})
                node_names.add(endpoint)
                stubs.append(endpoint)

    if stubs:
        print(f"  [AUTO-NODE] Created {len(stubs)} stub HDConcept node(s):")
        for s in stubs:
            print(f"             • {s!r}")

    return {"nodes": new_nodes, "edges": graph.get("edges", [])}


# ---------------------------------------------------------------------------
# Fix 2 — Cross-chunk node name normalisation
# ---------------------------------------------------------------------------

# Haiku sometimes prefixes RayJaiTeaching node names with the type name.
_TEACHING_PREFIX_RE = re.compile(r"^RayJaiTeaching:\s*", re.IGNORECASE)


def _normalise_name(name: str) -> str:
    """Strip whitespace and remove any spurious 'RayJaiTeaching: ' prefix."""
    name = name.strip()
    return _TEACHING_PREFIX_RE.sub("", name).strip()


def normalise_graph(graph: dict) -> dict:
    """
    One-pass normalisation of the fully-merged graph:

    1. Strip whitespace from every node name and edge endpoint.
    2. Remove the 'RayJaiTeaching: ' prefix from node names where present.
    3. Re-deduplicate nodes using case-insensitive name comparison, merging
       fields with the usual prefer-longer-string rule. Original casing of
       the first-seen form is preserved in the output.
    4. Update all edge endpoints to use the canonical (deduplicated) name.
    5. Re-deduplicate edges after endpoint canonicalisation.
    """
    # -- Step 1 & 2: normalise node names and build a rename map --
    rename: dict[str, str] = {}
    for node in graph.get("nodes", []):
        old = node.get("name") or ""
        new = _normalise_name(old)
        node["name"] = new
        if old != new:
            rename[old] = new

    # -- Step 3: case-insensitive node dedup (first-seen casing wins) --
    lower_to_node: dict[str, dict] = {}
    for node in graph.get("nodes", []):
        key = node["name"].lower()
        if key in lower_to_node:
            lower_to_node[key] = merge_node_fields(lower_to_node[key], node)
        else:
            lower_to_node[key] = node

    # canonical lookup: any casing → the preserved name stored in lower_to_node
    canonical: dict[str, str] = {
        key: node["name"] for key, node in lower_to_node.items()
    }

    # -- Step 4 & 5: fix edge endpoints and re-dedup --
    deduped_edges: list[dict] = []
    seen_edges: set[tuple] = set()
    for edge in graph.get("edges", []):
        raw_from = (edge.get("from") or "").strip()
        raw_to = (edge.get("to") or "").strip()
        # Apply prefix-removal rename if applicable, then canonicalise casing.
        raw_from = rename.get(raw_from, raw_from)
        raw_to = rename.get(raw_to, raw_to)
        edge["from"] = canonical.get(raw_from.lower(), raw_from)
        edge["to"] = canonical.get(raw_to.lower(), raw_to)

        key = (edge["from"], edge.get("type"), edge["to"])
        if key not in seen_edges:
            seen_edges.add(key)
            deduped_edges.append(edge)

    return {"nodes": list(lower_to_node.values()), "edges": deduped_edges}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_model(chunk_body: str) -> dict:
    """
    Send one chunk to the local Ollama model and return the parsed
    {nodes, edges} dict. Raises ValueError if the response is not valid JSON.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(chunk_body=chunk_body)
    raw = chat_json(SYSTEM_PROMPT, user_prompt)
    return extract_json(raw)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    print(f"Extraction model (Ollama): {MODEL}\n")

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
            result = call_model(chunk["body"])
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

    # Post-processing: normalise names, then guarantee every edge has a node.
    print("\nNormalising graph...")
    graph = normalise_graph(graph)
    graph = ensure_edge_nodes(graph)

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
