"""
reader.py — The Reader

Takes a user question, generates a Cypher query via Claude, retrieves
relevant RayJai teachings from Neo4j, then synthesises a response in
RayJai's voice using Claude Sonnet.

Usage:
  python scripts/reader.py           # normal mode — response only
  python scripts/reader.py --debug   # prints query, teachings, saves log

Environment:
  ANTHROPIC_API_KEY — required (.env in project root is loaded automatically)
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

CYPHER_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"

CYPHER_SYSTEM = """\
You are a Cypher query generator for a Neo4j Human Design knowledge graph.

Node types: RayJaiTeaching, HDType, HDCenter, HDGate, HDChannel, HDProfile, HDAuthority, HDConcept, Person.

RayJaiTeaching fields: name, insight, context, related_hd_concept.

Valid relationship types ONLY: ILLUSTRATES, TEACHES_THROUGH, REFRAMES, GRANTS_PERMISSION, DEFINED_IN, CONNECTS_TO, FORMS, DEFINES, TYPE_OF, BELONGS_TO, ACTIVE_IN, CONDITIONING_OF, PART_OF, BUILDS_ON, RELATES_TO.

Do NOT use any relationship type not in that list.

For most questions, query RayJaiTeaching directly using WHERE clauses on insight, context, and related_hd_concept fields rather than traversing relationships.

Always RETURN r.name, r.insight, r.context, r.related_hd_concept.
Limit to 20 results.
Return ONLY the Cypher query. No explanation. No markdown. No backticks."""

SYNTHESIS_SYSTEM = """\
You are RayJai Babauta — a Human Design reader and teacher.
You are speaking directly to a client in a reading session.
Speak in first person as RayJai. Never refer to RayJai in third person.
Use RayJai's actual speech patterns from the teachings provided:
- Speak conversationally, like you are talking not writing
- Use short punchy sentences mixed with longer ones
- Use "you know", "right?", "yeah", "babe", "to your point" naturally
- Use analogies and metaphors — the turtle, the closed book, the battery
- Do NOT use headers, bullet points, bold text, or markdown of any kind
- Do NOT structure your response like an essay
- Speak with warmth and directness — like you know this person
- Grant permission naturally in your own words — not as a formal statement
- Keep it conversational — 150 to 250 words maximum
Answer using ONLY the teachings provided.
If the teachings don't contain enough to answer, say "I haven't spoken about this yet in our sessions.\""""

# ---------------------------------------------------------------------------
# Step 1 — Generate Cypher query
# ---------------------------------------------------------------------------

def generate_cypher(client: anthropic.Anthropic, question: str) -> str:
    message = client.messages.create(
        model=CYPHER_MODEL,
        max_tokens=512,
        system=CYPHER_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    # Strip any accidental fences or whitespace the model sneaks in.
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

# ---------------------------------------------------------------------------
# Step 2 — Run query against Neo4j
# ---------------------------------------------------------------------------

def first_meaningful_word(question: str) -> str:
    """Extract the first meaningful noun from the question for the fallback query."""
    stopwords = {
        "what", "does", "rayjai", "teach", "about", "the", "and", "why",
        "it", "how", "is", "are", "do", "a", "an", "for", "to", "of",
        "in", "on",
    }
    for word in re.findall(r"[a-zA-Z]+", question):
        if word.lower() not in stopwords:
            return word
    return question.split()[0] if question.split() else "design"


FALLBACK_QUERY = """\
MATCH (r:RayJaiTeaching)
WHERE toLower(r.insight) CONTAINS toLower($keyword)
RETURN r.name, r.insight, r.context, r.related_hd_concept
LIMIT 20"""


def run_query(driver: GraphDatabase.driver, cypher: str, question: str) -> list[dict]:
    with driver.session() as session:
        try:
            results = session.run(cypher).data()
            if results:
                return results
            # Query ran fine but returned nothing — try fallback.
            keyword = first_meaningful_word(question)
            print(f"  (no results — falling back to keyword search: {keyword!r})")
            return session.run(FALLBACK_QUERY, keyword=keyword).data()
        except Exception as exc:
            keyword = first_meaningful_word(question)
            print(f"  [QUERY ERROR] {exc}")
            print(f"  (falling back to keyword search: {keyword!r})")
            return session.run(FALLBACK_QUERY, keyword=keyword).data()

# ---------------------------------------------------------------------------
# Step 3 — Format teachings and synthesise response
# ---------------------------------------------------------------------------

def format_teachings(results: list[dict]) -> str:
    if not results:
        return "(no teachings found)"
    lines = []
    for i, row in enumerate(results, 1):
        name = row.get("r.name") or ""
        insight = row.get("r.insight") or ""
        context = row.get("r.context") or ""
        concept = row.get("r.related_hd_concept") or ""
        lines.append(f"{i}. [{name}]")
        if insight:
            lines.append(f"   Insight: {insight}")
        if context:
            lines.append(f"   Context: {context}")
        if concept:
            lines.append(f"   Concept: {concept}")
    return "\n".join(lines)


def synthesise(client: anthropic.Anthropic, question: str, teachings: str) -> str:
    user_content = (
        f"TEACHINGS FROM THE GRAPH:\n{teachings}\n\n"
        f"USER QUESTION: {question}"
    )
    message = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=1024,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text.strip()

# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

_LOG_FILE = _PROJECT_ROOT / "output" / "reader_log.jsonl"


def log_interaction(
    question: str,
    cypher: str,
    results: list[dict],
    response: str,
) -> None:
    """Append one JSON object to the JSONL log file."""
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "cypher_query": cypher,
        "teachings_retrieved": len(results),
        "teachings": [
            {
                "name": r.get("r.name") or "",
                "insight": r.get("r.insight") or "",
            }
            for r in results
        ],
        "response": response,
    }
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    debug = "--debug" in sys.argv

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("The Reader — Ask anything about Human Design")
    if debug:
        print(f"[debug mode — logging to {_LOG_FILE}]")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break

            # Step 1 — Generate Cypher
            if debug:
                print("\nGenerating query...")
            try:
                cypher = generate_cypher(client, question)
            except Exception as exc:
                print(f"[ERROR generating query] {exc}")
                continue

            if debug:
                print(f"\nCypher:\n{cypher}\n")

            # Step 2 — Run query
            results = run_query(driver, cypher, question)

            if debug:
                print(f"Retrieved {len(results)} teaching(s).\n")
                if results:
                    print("Teachings:")
                    for i, row in enumerate(results, 1):
                        name = row.get("r.name") or ""
                        insight = row.get("r.insight") or ""
                        print(f"  {i}. [{name}] {insight[:120]}{'...' if len(insight) > 120 else ''}")
                    print()

            # Step 3 — Synthesise
            teachings = format_teachings(results)
            try:
                response = synthesise(client, question, teachings)
            except Exception as exc:
                print(f"[ERROR synthesising response] {exc}")
                continue

            print(f"\n{response}\n")
            print("-" * 60)

            if debug:
                log_interaction(question, cypher, results, response)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
