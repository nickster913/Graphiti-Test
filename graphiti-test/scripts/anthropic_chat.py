"""
anthropic_chat.py

Shared Anthropic Claude API helper for every LLM call in this project
(entity/relationship extraction and answer synthesis). Centralises client
creation, API-key handling, model selection, and JSON parsing so all scripts
call Claude the same, correct way.

Models (Claude API IDs, current as of Sep 2026). Override any via env var:
  ANTHROPIC_EXTRACT_MODEL    default claude-sonnet-5   (structured extraction)
  ANTHROPIC_SYNTHESIS_MODEL  default claude-sonnet-5   (voice synthesis)
  ANTHROPIC_CYPHER_MODEL     default claude-haiku-4-5  (cheap Cypher fallback)
  ANTHROPIC_EVAL_MODEL       default claude-haiku-4-5  (LLM-as-judge for evals)

Available tiers if you want to trade cost/quality:
  claude-opus-5      — strongest, pricier
  claude-sonnet-5    — best balance of speed + intelligence (default)
  claude-haiku-4-5   — fastest / cheapest

Requires ANTHROPIC_API_KEY. Callers load the project .env via python-dotenv
before the first call.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

EXTRACT_MODEL = os.getenv("ANTHROPIC_EXTRACT_MODEL", "claude-sonnet-5")
SYNTHESIS_MODEL = os.getenv("ANTHROPIC_SYNTHESIS_MODEL", "claude-sonnet-5")
CYPHER_MODEL = os.getenv("ANTHROPIC_CYPHER_MODEL", "claude-haiku-4-5")
EVAL_MODEL = os.getenv("ANTHROPIC_EVAL_MODEL", "claude-haiku-4-5")

# Strips markdown code fences if the model wraps JSON in them despite instructions.
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Return a cached Anthropic client, or exit clearly if the key is missing."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Add it to the project .env (ANTHROPIC_API_KEY=sk-ant-...) or export it."
        )
    return anthropic.Anthropic(api_key=api_key)


def _usage_details(message) -> dict[str, int] | None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    details: dict[str, int] = {
        "input": int(getattr(usage, "input_tokens", 0) or 0),
        "output": int(getattr(usage, "output_tokens", 0) or 0),
    }
    cached = getattr(usage, "cache_read_input_tokens", None)
    if cached:
        details["cache_read_input_tokens"] = int(cached)
    return details


def complete(
    system: str,
    user: str,
    *,
    model: str,
    max_tokens: int = 4096,
    name: str = "claude-complete",
) -> str:
    """Single-turn Claude call → assistant text (all text blocks concatenated).

    Recorded in Langfuse as a `generation` when tracing is configured.
    `name` is a stable observation name (used by evaluators / dashboards).
    """
    from langfuse_tracing import get_langfuse

    with get_langfuse().start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        },
        model_parameters={"max_tokens": max_tokens},
    ) as generation:
        message = get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
        generation.update(output=text, usage_details=_usage_details(message))
        return text


def complete_json(
    system: str,
    user: str,
    *,
    model: str = EXTRACT_MODEL,
    max_tokens: int = 4096,
    name: str = "extract-entities",
) -> dict:
    """Claude call that must return one JSON object; parses and returns it."""
    raw = complete(system, user, model=model, max_tokens=max_tokens, name=name)
    text = raw.strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)
