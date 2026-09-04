"""
reader.py — The Reader (hybrid RAG)

Takes a user question and answers it in RayJai's voice using a hybrid
retrieval-augmented pipeline over the Neo4j knowledge graph:

  1. ANCHOR  — embed the question (Ollama nomic-embed-text) and vector-search
               RayJaiTeaching nodes for the most semantically relevant teachings.
  2. EXPAND  — traverse the graph from those anchors (shared related_hd_concept)
               to gather connected teachings for multi-hop context.
  3. VOICE   — vector-search HDVoicePattern nodes for verbatim voice exemplars
               matching the question (falls back to keyword behaviour context).
  4. SYNTHESISE — Claude Sonnet composes the reply, using the teachings as
               content and the exemplars to mirror RayJai's phrasing.

If embeddings/vector indexes are not present (embed_graph.py not run) or Ollama
is unreachable, retrieval gracefully falls back to the previous behaviour:
Claude-generated Cypher, then a keyword CONTAINS search.

Usage:
  python scripts/reader.py           # normal mode — response only
  python scripts/reader.py --debug   # prints retrieval details, saves a log

Environment:
  ANTHROPIC_API_KEY       — required (.env in project root is loaded automatically)
  LANGFUSE_PUBLIC_KEY     — optional; with LANGFUSE_SECRET_KEY, each question is
  LANGFUSE_SECRET_KEY       traced to local Langfuse (embed → retrieve → synthesise)
  LANGFUSE_BASE_URL       — default http://localhost:3100 (docker-compose.langfuse.yml)

Prerequisites for full RAG mode:
  - Run scripts/load_graph.py then scripts/embed_graph.py
  - Ollama running with nomic-embed-text pulled
"""

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from anthropic_chat import CYPHER_MODEL, SYNTHESIS_MODEL, complete, get_client
from embed_utils import EMBED_MODEL, embed_text
from langfuse_tracing import flush, get_langfuse, observe, propagate_attributes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Model IDs are centralised in anthropic_chat.py (CYPHER_MODEL, SYNTHESIS_MODEL).

TEACHING_INDEX = "rayjai_teaching_embedding"
VOICE_INDEX = "hdvoice_embedding"

# Retrieval sizing
ANCHOR_K = 8        # vector hits used as anchors
EXPAND_LIMIT = 6    # extra teachings pulled in by graph expansion
VOICE_K = 5         # voice exemplars
MAX_TEACHINGS = 12  # cap sent to synthesis

CYPHER_SYSTEM = """\
You are a Cypher query generator for a Neo4j Human Design knowledge graph.

Node types: RayJaiTeaching, HDVoicePattern, HDSessionFlow, HDBehaviourRule, HDToneProfile, HDType, HDCenter, HDGate, HDChannel, HDProfile, HDAuthority, HDConcept, Person.

RayJaiTeaching fields: name, insight, context, related_hd_concept, trigger_context, effect.

Valid relationship types ONLY: ILLUSTRATES, TEACHES_THROUGH, REFRAMES, GRANTS_PERMISSION, DEFINED_IN, CONNECTS_TO, FORMS, DEFINES, TYPE_OF, BELONGS_TO, ACTIVE_IN, CONDITIONING_OF, PART_OF, BUILDS_ON, RELATES_TO, EXPRESSES, STEP_OF, GOVERNS, CALIBRATES_FOR.

For most questions, query RayJaiTeaching directly using WHERE clauses on insight, context, and related_hd_concept fields.
Always RETURN r.name, r.insight, r.context, r.related_hd_concept.
Limit to 20 results.
Return ONLY the Cypher query. No explanation. No markdown. No backticks.
Match on related_hd_concept and context using broad concept terms extracted from the question (e.g. "Manifestor", "Generator", "authority", "open center").
For questions about feelings (e.g. "too intense", "always angry", "can't decide"), map to the relevant HD concept: intensity/anger → Manifestor not-self, can't decide → authority, tired/exhausted → Generator or open Sacral, etc."""

SYNTHESIS_SYSTEM = """\
You are RayJai Babauta — a Human Design reader and teacher.
You are speaking directly to a client in a reading session.
Speak in first person as RayJai. Never refer to RayJai in third person.
Use the teachings provided in TEACHINGS FROM THE GRAPH as the CONTENT of your response.
Use the voice patterns, behaviour rules, and tone calibration in BEHAVIOUR CONTEXT to guide HOW you speak.
Mirror the phrasing, rhythm, and characteristic word choices shown in the exemplars — reuse his expressions where they fit naturally, rather than inventing your own style.
Do NOT use headers, bullet points, bold text, or markdown of any kind.
Keep it conversational — 150 to 250 words maximum.
Answer using ONLY the teachings and context provided.
If the teachings don't contain enough to answer, say "I haven't spoken about this yet in our sessions.\""""

# ---------------------------------------------------------------------------
# Question embedding (graceful if Ollama is down)
# ---------------------------------------------------------------------------

@observe(name="embed-question", as_type="embedding", capture_input=False, capture_output=False)
def safe_embed(question: str, debug: bool) -> list[float] | None:
    """Embed the question, or return None (so retrieval falls back to keyword)."""
    lf = get_langfuse()
    lf.update_current_generation(input=question, model=EMBED_MODEL)
    try:
        vector = embed_text(question)
        lf.update_current_generation(output={"dimensions": len(vector)})
        return vector
    except Exception as exc:  # noqa: BLE001
        lf.update_current_generation(
            output={"error": str(exc)},
            metadata={"fallback": "keyword"},
        )
        if debug:
            print(f"  [embed unavailable — falling back to keyword search] {exc}")
        return None

# ---------------------------------------------------------------------------
# Legacy fallback — Claude-generated Cypher + keyword search
# ---------------------------------------------------------------------------

def generate_cypher(question: str) -> str:
    raw = complete(
        CYPHER_SYSTEM,
        question,
        model=CYPHER_MODEL,
        max_tokens=512,
        name="generate-cypher",
    )
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def first_meaningful_word(question: str) -> str:
    """Extract the first meaningful noun from the question for keyword fallback."""
    stopwords = {
        "what", "does", "rayjai", "teach", "about", "the", "and", "why",
        "it", "how", "is", "are", "do", "a", "an", "for", "to", "of",
        "in", "on", "i", "i'm", "im", "my", "me", "we", "us", "you",
        "your", "this", "that", "keep", "getting", "told", "always",
        "never", "just", "really", "so", "very", "too", "been", "have",
        "has", "was", "were", "be", "am",
    }
    words = re.findall(r"[a-zA-Z]+", question)
    for word in words:
        if word.lower() not in stopwords and len(word) > 2:
            return word
    return words[0] if words else "design"


FALLBACK_QUERY = """\
MATCH (r:RayJaiTeaching)
WHERE toLower(r.insight) CONTAINS toLower($keyword)
RETURN r.name AS name, r.insight AS insight, r.context AS context,
       r.related_hd_concept AS related
LIMIT 20"""


def legacy_content(driver, question, debug) -> list[dict]:
    """Old path: LLM-generated Cypher, then keyword CONTAINS fallback."""
    try:
        cypher = generate_cypher(question)
        if debug:
            print(f"  [fallback Cypher]\n{cypher}\n")
        with driver.session() as session:
            rows = session.run(cypher).data()
        rows = [_normalise_legacy_row(r) for r in rows]
        rows = [r for r in rows if r["name"] or r["insight"]]
        if rows:
            return rows
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(f"  [fallback Cypher error] {exc}")

    keyword = first_meaningful_word(question)
    if debug:
        print(f"  [keyword fallback: {keyword!r}]")
    with driver.session() as session:
        rows = session.run(FALLBACK_QUERY, keyword=keyword).data()
    return [_normalise_legacy_row(r) for r in rows]


def _normalise_legacy_row(row: dict) -> dict:
    """Accept either aliased (name/insight/...) or r.-prefixed keys."""
    return {
        "name": row.get("name") or row.get("r.name") or "",
        "insight": row.get("insight") or row.get("r.insight") or "",
        "context": row.get("context") or row.get("r.context") or "",
        "related": row.get("related") or row.get("r.related_hd_concept") or "",
    }

# ---------------------------------------------------------------------------
# Hybrid retrieval — vector anchor + graph expansion
# ---------------------------------------------------------------------------

VECTOR_TEACHING_QUERY = f"""
CALL db.index.vector.queryNodes('{TEACHING_INDEX}', $k, $vec)
YIELD node, score
RETURN node.name AS name, node.insight AS insight, node.context AS context,
       node.related_hd_concept AS related, score
"""

EXPAND_QUERY = """
UNWIND $concepts AS concept
MATCH (r:RayJaiTeaching)
WHERE r.related_hd_concept = concept AND NOT r.name IN $seen
RETURN DISTINCT r.name AS name, r.insight AS insight, r.context AS context,
       r.related_hd_concept AS related
LIMIT $limit
"""


@observe(name="retrieve-content", as_type="retriever", capture_input=False, capture_output=False)
def retrieve_content(driver, question, qvec, debug) -> tuple[list[dict], str]:
    """
    Return (teachings, mode). mode is "vector" for RAG or "keyword" for fallback.
    """
    lf = get_langfuse()
    lf.update_current_span(input={"question": question})

    if qvec is not None:
        try:
            with driver.session() as session:
                anchors = session.run(
                    VECTOR_TEACHING_QUERY, k=ANCHOR_K, vec=qvec
                ).data()

                seen = [a["name"] for a in anchors if a.get("name")]
                concepts = sorted(
                    {a["related"] for a in anchors if a.get("related")}
                )
                expanded = []
                if concepts:
                    expanded = session.run(
                        EXPAND_QUERY,
                        concepts=concepts,
                        seen=seen,
                        limit=EXPAND_LIMIT,
                    ).data()

            rows = _dedupe_by_name(anchors + expanded)[:MAX_TEACHINGS]
            rows = [_strip_score(r) for r in rows]
            if rows:
                lf.update_current_span(
                    output=_retrieval_output(rows, "vector"),
                    metadata={"mode": "vector", "anchor_k": ANCHOR_K},
                )
                return rows, "vector"
            if debug:
                print("  [vector search returned nothing — using fallback]")
        except Exception as exc:  # noqa: BLE001
            if debug:
                print(f"  [vector search unavailable — using fallback] {exc}")

    rows = legacy_content(driver, question, debug)
    lf.update_current_span(
        output=_retrieval_output(rows, "keyword"),
        metadata={"mode": "keyword"},
    )
    return rows, "keyword"


def _retrieval_output(rows: list[dict], mode: str) -> dict:
    return {
        "mode": mode,
        "count": len(rows),
        "names": [r.get("name") or "" for r in rows],
    }


def _dedupe_by_name(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        name = r.get("name") or ""
        if name in seen:
            continue
        seen.add(name)
        out.append(r)
    return out


def _strip_score(row: dict) -> dict:
    return {
        "name": row.get("name") or "",
        "insight": row.get("insight") or "",
        "context": row.get("context") or "",
        "related": row.get("related") or "",
    }

# ---------------------------------------------------------------------------
# Voice retrieval — vector exemplars, keyword fallback
# ---------------------------------------------------------------------------

VECTOR_VOICE_QUERY = f"""
CALL db.index.vector.queryNodes('{VOICE_INDEX}', $k, $vec)
YIELD node, score
RETURN 'HDVoicePattern' AS node_type, node.name AS name,
       node.phrase AS content, node.usage_context AS context, score
"""

BEHAVIOUR_QUERY = """\
MATCH (v:HDVoicePattern)
WHERE toLower(v.usage_context) CONTAINS toLower($keyword)
   OR toLower(v.phrase) CONTAINS toLower($keyword)
RETURN 'HDVoicePattern' AS node_type, v.name AS name, v.phrase AS content, v.usage_context AS context
LIMIT 5

UNION

MATCH (b:HDBehaviourRule)
WHERE b.rule_type = 'DO' OR toLower(b.context) CONTAINS toLower($keyword)
RETURN 'HDBehaviourRule' AS node_type, b.name AS name, b.rule AS content, b.context AS context
LIMIT 5

UNION

MATCH (t:HDToneProfile)
WHERE toLower(t.emotional_state) CONTAINS toLower($keyword)
RETURN 'HDToneProfile' AS node_type, t.name AS name, t.instruction AS content, t.emotional_state AS context
LIMIT 3"""


@observe(name="retrieve-voice", as_type="retriever", capture_input=False, capture_output=False)
def retrieve_voice(driver, question, qvec, debug) -> tuple[list[dict], str]:
    lf = get_langfuse()
    lf.update_current_span(input={"question": question})

    if qvec is not None:
        try:
            with driver.session() as session:
                rows = session.run(VECTOR_VOICE_QUERY, k=VOICE_K, vec=qvec).data()
            rows = [
                {
                    "node_type": r.get("node_type") or "HDVoicePattern",
                    "name": r.get("name") or "",
                    "content": r.get("content") or "",
                    "context": r.get("context") or "",
                }
                for r in rows
            ]
            if rows:
                lf.update_current_span(
                    output=_retrieval_output(rows, "vector"),
                    metadata={"mode": "vector"},
                )
                return rows, "vector"
        except Exception as exc:  # noqa: BLE001
            if debug:
                print(f"  [voice vector search unavailable — using fallback] {exc}")

    keyword = first_meaningful_word(question)
    try:
        with driver.session() as session:
            rows = session.run(BEHAVIOUR_QUERY, keyword=keyword).data()
        lf.update_current_span(
            output=_retrieval_output(rows, "keyword"),
            metadata={"mode": "keyword", "keyword": keyword},
        )
        return rows, "keyword"
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(f"  [behaviour query error] {exc}")
        lf.update_current_span(output=_retrieval_output([], "keyword"))
        return [], "keyword"

# ---------------------------------------------------------------------------
# Formatting + synthesis
# ---------------------------------------------------------------------------

def format_teachings(results: list[dict]) -> str:
    if not results:
        return "(no teachings found)"
    lines = []
    for i, row in enumerate(results, 1):
        lines.append(f"{i}. [{row.get('name', '')}]")
        if row.get("insight"):
            lines.append(f"   Insight: {row['insight']}")
        if row.get("context"):
            lines.append(f"   Context: {row['context']}")
        if row.get("related"):
            lines.append(f"   Concept: {row['related']}")
    return "\n".join(lines)


def format_behaviour_context(results: list[dict]) -> str:
    if not results:
        return "(no behaviour context found)"
    lines = []
    for i, row in enumerate(results, 1):
        lines.append(f"{i}. [{row.get('node_type', '')}] {row.get('name', '')}")
        if row.get("content"):
            lines.append(f"   Rule/Pattern: {row['content']}")
        if row.get("context"):
            lines.append(f"   Context: {row['context']}")
    return "\n".join(lines)


def synthesise(question, teachings, behaviour_context) -> str:
    user_content = (
        f"BEHAVIOUR CONTEXT FROM GRAPH:\n{behaviour_context}\n\n"
        f"TEACHINGS FROM THE GRAPH:\n{teachings}\n\n"
        f"USER QUESTION: {question}"
    )
    return complete(
        SYNTHESIS_SYSTEM,
        user_content,
        model=SYNTHESIS_MODEL,
        max_tokens=1024,
        name="synthesise-answer",
    )

# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

_LOG_FILE = _PROJECT_ROOT / "output" / "reader_log.jsonl"


def log_interaction(question, retrieval_mode, results, response) -> None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieval_mode": retrieval_mode,
        "teachings_retrieved": len(results),
        "teachings": [
            {"name": r.get("name", ""), "insight": r.get("insight", "")}
            for r in results
        ],
        "response": response,
    }
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# One-turn pipeline (used by the interactive loop and eval_reader.py)
# ---------------------------------------------------------------------------

@observe(name="answer-question", capture_input=False, capture_output=False)
def answer_question(driver, question: str, *, debug: bool = False) -> dict:
    """
    Run the hybrid RAG pipeline for one question.

    Returns a dict with answer, retrieved teachings/voice, and retrieval modes
    so Langfuse evaluators can score faithfulness against the actual context.
    """
    lf = get_langfuse()
    lf.update_current_span(input={"question": question})

    qvec = safe_embed(question, debug)
    results, content_mode = retrieve_content(driver, question, qvec, debug)
    behaviour_results, voice_mode = retrieve_voice(driver, question, qvec, debug)

    teachings = format_teachings(results)
    behaviour_context = format_behaviour_context(behaviour_results)
    response = synthesise(question, teachings, behaviour_context)

    lf.update_current_span(
        output={"answer": response},
        metadata={
            "content_mode": content_mode,
            "voice_mode": voice_mode,
            "teaching_count": str(len(results)),
            "voice_count": str(len(behaviour_results)),
        },
    )
    return {
        "answer": response,
        "teachings": results,
        "voice": behaviour_results,
        "content_mode": content_mode,
        "voice_mode": voice_mode,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    debug = "--debug" in sys.argv

    get_client()  # fail fast if ANTHROPIC_API_KEY is missing
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    session_id = str(uuid.uuid4())

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

            try:
                with propagate_attributes(
                    session_id=session_id,
                    tags=["reader"],
                    metadata={"app": "graphiti-test"},
                ):
                    result = answer_question(driver, question, debug=debug)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR synthesising response] {exc}")
                continue

            results = result["teachings"]
            behaviour_results = result["voice"]
            content_mode = result["content_mode"]
            voice_mode = result["voice_mode"]
            response = result["answer"]

            if debug:
                print(f"\nRetrieval: content={content_mode}, voice={voice_mode}")
                print(f"Retrieved {len(results)} teaching(s), "
                      f"{len(behaviour_results)} voice exemplar(s).\n")
                for i, row in enumerate(results, 1):
                    insight = row.get("insight", "")
                    print(f"  {i}. [{row.get('name', '')}] "
                          f"{insight[:120]}{'...' if len(insight) > 120 else ''}")
                print()

            print(f"\n{response}\n")
            print("-" * 60)

            if debug:
                log_interaction(question, f"{content_mode}/{voice_mode}", results, response)
            flush()
    finally:
        flush()
        driver.close()


if __name__ == "__main__":
    main()
