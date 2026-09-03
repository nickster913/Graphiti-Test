"""
extract_training_doc.py

One-shot extraction script for The 26•44 Agent Training document.
Reads the raw training document, sends it to the Claude API in sections,
and outputs a structured JSON file to output/training_graph_seed.json.

Usage:
  uv run python scripts/extract_training_doc.py

Environment:
  ANTHROPIC_API_KEY        — required (.env in project root is loaded automatically)
  ANTHROPIC_EXTRACT_MODEL  — optional, default claude-sonnet-5 (see scripts/anthropic_chat.py)

Output:
  output/training_graph_seed.json
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from anthropic_chat import EXTRACT_MODEL, complete_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DOC = _PROJECT_ROOT / "transcripts" / "The2644 AgentTraining_v1.docx.txt"
OUTPUT_FILE = _PROJECT_ROOT / "output" / "training_graph_seed.json"
MODEL = EXTRACT_MODEL
MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0  # seconds (SDK retries handle transient 429s)
SECTION_SEPARATOR = "________________"

SYSTEM_PROMPT = """\
You are a knowledge graph extraction specialist for the 26•44 Human Design platform.
Extract agent behaviour nodes and relationships from this training document section.
Return ONLY valid JSON. No preamble, no explanation, no markdown code blocks.
The JSON must follow this exact schema:
{"nodes": [...], "edges": [...]}"""

USER_PROMPT_TEMPLATE = """\
Extract all agent behaviour entities from this section of the training document.

NODE TYPES AND FIELDS:

HDVoicePattern: name (short label), phrase (the exact phrase or pattern), usage_context (when to use it), trigger (what situation triggers it), avoid_if (when NOT to use it)
HDSessionFlow: name (short label), step_number (integer), instruction (what to do), purpose (why), example_variant (an acceptable variation)
HDBehaviourRule: name (short label), rule_type ("DO" or "DON'T"), rule (the rule itself), context (when it applies), rationale (why this rule exists)
HDToneProfile: name (short label), emotional_state (what state the user is in), instruction (how to respond), example_response (an example of this tone in action)
RayJaiTeaching: name (short label), insight (RayJai's exact words where available), context (when he says this), related_hd_concept, trigger_context (emotional/situational trigger), effect (what the phrasing achieves: normalise/grant permission/reframe shame/etc)

EDGE TYPES:

EXPRESSES: HDVoicePattern → HDType or HDConcept
STEP_OF: HDSessionFlow → HDSessionFlow
GOVERNS: HDBehaviourRule → HDType or HDConcept or HDToneProfile
CALIBRATES_FOR: HDToneProfile → HDType or Person
ILLUSTRATES: RayJaiTeaching → any HD node
REFRAMES: RayJaiTeaching → HDConcept
GRANTS_PERMISSION: RayJaiTeaching → Person or HDType

RULES:

Every node must have "type" and "name" fields
Every edge must have "from", "to", and "type" fields
For HDVoicePattern: capture exact phrases from the document where they exist
For HDSessionFlow: step_number must be an integer matching the session flow order
For HDBehaviourRule: rule_type must be exactly "DO" or "DON'T"
Only extract what is explicitly in the section — do not invent
If nothing relevant is in the section, return {{"nodes": [], "edges": []}}

TRAINING DOCUMENT SECTION:
{section_body}"""

# ---------------------------------------------------------------------------
# Merge logic (copied from extract_nodes.py — not imported to keep standalone)
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
            if len(val) > len(merged[key]):
                merged[key] = val
    return merged


def merge_graphs(base: dict, incoming: dict) -> dict:
    """
    Merge incoming nodes/edges into base, deduplicating as described.
    """
    node_map: dict[str, dict] = {n["name"]: n for n in base.get("nodes", [])}
    for node in incoming.get("nodes", []):
        name = node.get("name")
        if not name:
            continue
        if name in node_map:
            node_map[name] = merge_node_fields(node_map[name], node)
        else:
            node_map[name] = node

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
       fields with the usual prefer-longer-string rule.
    4. Update all edge endpoints to use the canonical (deduplicated) name.
    5. Re-deduplicate edges after endpoint canonicalisation.
    """
    rename: dict[str, str] = {}
    for node in graph.get("nodes", []):
        old = node.get("name") or ""
        new = _normalise_name(old)
        node["name"] = new
        if old != new:
            rename[old] = new

    lower_to_node: dict[str, dict] = {}
    for node in graph.get("nodes", []):
        key = node["name"].lower()
        if key in lower_to_node:
            lower_to_node[key] = merge_node_fields(lower_to_node[key], node)
        else:
            lower_to_node[key] = node

    canonical: dict[str, str] = {
        key: node["name"] for key, node in lower_to_node.items()
    }

    deduped_edges: list[dict] = []
    seen_edges: set[tuple] = set()
    for edge in graph.get("edges", []):
        raw_from = (edge.get("from") or "").strip()
        raw_to = (edge.get("to") or "").strip()
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

def call_model(section_body: str) -> dict:
    """
    Send one section to Claude and return the parsed {nodes, edges} dict.
    Raises ValueError if the response is not valid JSON.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(section_body=section_body)
    return complete_json(SYSTEM_PROMPT, user_prompt, model=MODEL, max_tokens=MAX_TOKENS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Error: ANTHROPIC_API_KEY is not set. Add it to your .env.")
    print(f"Extraction model (Claude): {MODEL}\n")

    if not TRAINING_DOC.exists():
        raise SystemExit(
            f"Error: {TRAINING_DOC} not found.\n"
            "Place the training document at data/The2644_AgentTraining_v1.txt"
        )

    raw_text = TRAINING_DOC.read_text(encoding="utf-8")
    sections = [s.strip() for s in raw_text.split(SECTION_SEPARATOR) if s.strip()]

    print(f"Training document loaded: {len(sections)} section(s) found.\n")

    graph: dict = {"nodes": [], "edges": []}
    total = len(sections)
    skipped = 0

    for i, section_body in enumerate(sections):
        print(
            f"[{i + 1}/{total}] Extracting section {i + 1} "
            f"({len(section_body):,} chars)...",
            end=" ",
            flush=True,
        )

        try:
            result = call_model(section_body)
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

    print("\nNormalising graph...")
    graph = normalise_graph(graph)
    graph = ensure_edge_nodes(graph)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    nodes_total = len(graph["nodes"])
    edges_total = len(graph["edges"])

    print()
    print("=" * 50)
    print("TRAINING EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Sections processed : {total - skipped}/{total}  ({skipped} skipped)")
    print(f"Total nodes        : {nodes_total}")
    print(f"Total edges        : {edges_total}")
    print(f"Output written to  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
