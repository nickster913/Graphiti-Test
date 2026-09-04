"""
eval_reader.py — offline evaluation of the Reader pipeline via Langfuse.

Runs the gold questions in eval/reader_dataset.json through answer_question(),
scores each turn (deterministic checks + optional LLM-as-judge), and records
a dataset experiment in Langfuse when API keys are set.

Usage:
  uv run python scripts/eval_reader.py
  uv run python scripts/eval_reader.py --no-judge          # cheap: skip LLM judges
  uv run python scripts/eval_reader.py --limit 3           # first N items
  uv run python scripts/eval_reader.py --concurrency 1

Prerequisites:
  - Neo4j loaded (load_graph.py) and preferably embedded (embed_graph.py)
  - ANTHROPIC_API_KEY
  - Local Langfuse (docker compose -f docker-compose.langfuse.yml up -d)
    plus the keys in .env.example to record a dataset run
    (without them, scores still print locally)

Environment:
  ANTHROPIC_EVAL_MODEL  — judge model, default claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from anthropic_chat import EVAL_MODEL, SYNTHESIS_MODEL, complete_json, get_client
from langfuse_tracing import (
    Evaluation,
    flush,
    get_langfuse,
    tracing_configured,
)
from reader import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, answer_question

_PROJECT_ROOT = Path(__file__).parent.parent
DATASET_FILE = _PROJECT_ROOT / "eval" / "reader_dataset.json"
DATASET_NAME = "reader-eval"

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_MARKDOWN_RE = re.compile(
    r"(?m)^#{1,6}\s|\*\*[^*]+\*\*|__[^_]+__|^\s*[-*]\s|^\s*\d+\.\s",
)

JUDGE_SYSTEM = """\
You are a strict evaluator of a Human Design reading assistant.
Return ONLY valid JSON with this exact schema:
{"score": <float between 0 and 1>, "comment": "<one or two sentences>"}
Do not add markdown. Be harsh: 1.0 is rare."""


def _item_get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _answer_text(output) -> str:
    if isinstance(output, dict):
        return str(output.get("answer") or "")
    return str(output or "")


def _teachings_blob(output) -> str:
    if not isinstance(output, dict):
        return ""
    parts = []
    for row in output.get("teachings") or []:
        parts.append(row.get("name") or "")
        parts.append(row.get("insight") or "")
        parts.append(row.get("related") or "")
    return "\n".join(parts)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def word_count_in_range(*, output, **_kwargs) -> Evaluation:
    """Synthesis prompt asks for 150–250 words; score softly around that band."""
    n = _word_count(_answer_text(output))
    if 120 <= n <= 280:
        value = 1.0
        comment = f"{n} words — inside the target band"
    elif 80 <= n <= 350:
        value = 0.5
        comment = f"{n} words — close to 150–250"
    else:
        value = 0.0
        comment = f"{n} words — outside 150–250"
    return Evaluation(name="word_count_in_range", value=value, comment=comment)


def no_markdown(*, output, **_kwargs) -> Evaluation:
    text = _answer_text(output)
    hits = _MARKDOWN_RE.findall(text)
    ok = len(hits) == 0
    return Evaluation(
        name="no_markdown",
        value=1.0 if ok else 0.0,
        comment="clean prose" if ok else f"markdown-like patterns: {len(hits)}",
    )


def first_person_voice(*, output, **_kwargs) -> Evaluation:
    text = _answer_text(output)
    third_person = bool(re.search(r"\bRayJai\b", text, re.IGNORECASE))
    first = bool(re.search(r"\bI\b|\bmy\b|\bI'm\b|\bI've\b", text))
    if third_person:
        return Evaluation(
            name="first_person_voice",
            value=0.0,
            comment="refers to RayJai in third person",
        )
    if first:
        return Evaluation(
            name="first_person_voice",
            value=1.0,
            comment="speaks in first person",
        )
    return Evaluation(
        name="first_person_voice",
        value=0.5,
        comment="no third-person leak, but little first-person voice",
    )


def concept_hit(*, output, expected_output, metadata, **_kwargs) -> Evaluation:
    """Fraction of expected_concepts that appear in the answer or retrieved teachings."""
    meta = metadata or {}
    if not isinstance(meta, dict):
        meta = {}
    concepts = meta.get("expected_concepts") or []
    if isinstance(concepts, str):
        concepts = [concepts]
    if not concepts and expected_output:
        concepts = [w for w in str(expected_output).split() if len(w) > 4][:4]
    if not concepts:
        return Evaluation(name="concept_hit", value=None, comment="no expected concepts")

    blob = (_answer_text(output) + "\n" + _teachings_blob(output)).lower()
    hits = [c for c in concepts if str(c).lower() in blob]
    value = len(hits) / len(concepts)
    return Evaluation(
        name="concept_hit",
        value=round(value, 3),
        comment=f"{len(hits)}/{len(concepts)} concepts present: {hits}",
    )


def retrieval_mode_vector(*, output, **_kwargs) -> Evaluation:
    mode = output.get("content_mode") if isinstance(output, dict) else None
    ok = mode == "vector"
    return Evaluation(
        name="retrieval_mode_vector",
        value=1.0 if ok else 0.0,
        comment=f"content_mode={mode}",
    )


def _judge(name: str, user: str) -> Evaluation:
    try:
        data = complete_json(
            JUDGE_SYSTEM,
            user,
            model=EVAL_MODEL,
            max_tokens=256,
            name=f"judge-{name}",
        )
        score = float(data.get("score", 0))
        score = max(0.0, min(1.0, score))
        comment = str(data.get("comment") or "")
        return Evaluation(name=name, value=score, comment=comment)
    except Exception as exc:  # noqa: BLE001
        return Evaluation(name=name, value=None, comment=f"judge failed: {exc}")


def llm_faithfulness(*, input, output, **_kwargs) -> Evaluation:
    teachings = _teachings_blob(output) or "(none retrieved)"
    user = (
        f"Does the ANSWER stay faithful to the RETRIEVED TEACHINGS? "
        f"Penalise invented HD claims that are not supported by the teachings. "
        f"It is OK to say the sessions have not covered a topic.\n\n"
        f"QUESTION:\n{input}\n\n"
        f"RETRIEVED TEACHINGS:\n{teachings}\n\n"
        f"ANSWER:\n{_answer_text(output)}"
    )
    return _judge("faithfulness", user)


def llm_voice_fidelity(*, output, **_kwargs) -> Evaluation:
    user = (
        "Does this sound like RayJai speaking in a live reading? "
        "First person, conversational, no markdown/headers/bullets, "
        "permission-giving rather than textbook. Score 0 if it reads like a chatbot.\n\n"
        f"ANSWER:\n{_answer_text(output)}"
    )
    return _judge("voice_fidelity", user)


def llm_relevance(*, input, output, expected_output, **_kwargs) -> Evaluation:
    hint = expected_output or "(no gold hint)"
    user = (
        "Does the ANSWER actually address the QUESTION? "
        "Use the GOLD HINT as a coverage check, not a wording template.\n\n"
        f"QUESTION:\n{input}\n\n"
        f"GOLD HINT:\n{hint}\n\n"
        f"ANSWER:\n{_answer_text(output)}"
    )
    return _judge("relevance", user)


def _mean_run_evaluator(metric: str):
    def _avg(*, item_results, **_kwargs) -> Evaluation:
        values = [
            ev.value
            for result in item_results
            for ev in (result.evaluations or [])
            if ev.name == metric and isinstance(ev.value, (int, float))
        ]
        if not values:
            return Evaluation(name=f"avg_{metric}", value=None)
        avg = sum(values) / len(values)
        return Evaluation(
            name=f"avg_{metric}",
            value=round(avg, 3),
            comment=f"n={len(values)}",
        )

    _avg.__name__ = f"avg_{metric}"
    return _avg


def load_local_items(limit: int | None) -> list[dict]:
    payload = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    items = []
    for raw in payload["items"]:
        items.append(
            {
                "input": raw["input"],
                "expected_output": raw.get("expected_output"),
                "metadata": raw.get("metadata") or {},
                "id": raw.get("id"),
            }
        )
    if limit is not None:
        items = items[:limit]
    return items


def upsert_langfuse_dataset(items: list[dict]) -> None:
    lf = get_langfuse()
    try:
        lf.get_dataset(DATASET_NAME)
    except Exception:
        lf.create_dataset(
            name=DATASET_NAME,
            description=(
                "Hybrid RAG Reader gold questions for the Human Design graph. "
                "Used by scripts/eval_reader.py."
            ),
        )
    for item in items:
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item.get("id"),
            input=item["input"],
            expected_output=item.get("expected_output"),
            metadata=item.get("metadata"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Reader with Langfuse")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scores")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N items")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max parallel Reader calls (default 2)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = parse_args()

    if not DATASET_FILE.exists():
        raise SystemExit(f"Error: dataset not found at {DATASET_FILE}")

    get_client()  # fail fast without an Anthropic key
    local_items = load_local_items(args.limit)
    print(f"Evaluating {len(local_items)} item(s) from {DATASET_FILE.name}")
    print(f"Synthesis model: {SYNTHESIS_MODEL}")
    if args.no_judge:
        print("LLM judges: off")
    else:
        print(f"Judge model: {EVAL_MODEL}")

    if tracing_configured():
        print(f"Langfuse: uploading dataset {DATASET_NAME!r} and recording a run")
        upsert_langfuse_dataset(local_items)
    else:
        print(
            "Langfuse: keys not set — scores print locally, traces are not exported.\n"
            "Start docker compose -f docker-compose.langfuse.yml and copy the\n"
            "LANGFUSE_* lines from .env.example into .env."
        )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def task(*, item, **_kwargs):
        from langfuse_tracing import propagate_attributes

        question = _item_get(item, "input")
        with propagate_attributes(tags=["eval", "reader"]):
            return answer_question(driver, question, debug=False)

    evaluators = [
        word_count_in_range,
        no_markdown,
        first_person_voice,
        concept_hit,
        retrieval_mode_vector,
    ]
    if not args.no_judge:
        evaluators.extend([llm_faithfulness, llm_voice_fidelity, llm_relevance])

    run_evaluators = [
        _mean_run_evaluator(name)
        for name in (
            "word_count_in_range",
            "no_markdown",
            "first_person_voice",
            "concept_hit",
            "retrieval_mode_vector",
            *([] if args.no_judge else ["faithfulness", "voice_fidelity", "relevance"]),
        )
    ]

    lf = get_langfuse()
    experiment_kwargs = dict(
        name="Reader RAG eval",
        description="Hybrid retrieval + RayJai voice synthesis",
        task=task,
        evaluators=evaluators,
        run_evaluators=run_evaluators,
        max_concurrency=args.concurrency,
        metadata={
            "synthesis_model": SYNTHESIS_MODEL,
            "eval_model": EVAL_MODEL if not args.no_judge else "none",
        },
    )

    try:
        if tracing_configured():
            dataset = lf.get_dataset(DATASET_NAME)
            wanted = {item.get("id") for item in local_items if item.get("id")}
            data = [
                it for it in dataset.items
                if getattr(it, "id", None) in wanted
            ]
            result = lf.run_experiment(data=data or local_items, **experiment_kwargs)
        else:
            result = lf.run_experiment(data=local_items, **experiment_kwargs)
        print()
        print(result.format())
    finally:
        flush()
        driver.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
