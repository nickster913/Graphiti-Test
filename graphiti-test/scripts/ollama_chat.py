"""
ollama_chat.py

Local LLM chat helper for JSON extraction via Ollama (default: qwen3.6:27b).
Used by extract_nodes.py and extract_training_doc.py so entity/relationship
extraction runs on a local model instead of the Claude API.

No extra Python package is required — this calls Ollama's HTTP API via httpx.
Ollama's `format: "json"` constrains the output to a single JSON object, and
`think: false` disables Qwen's reasoning tokens so the content is clean JSON.

Environment (optional overrides):
  OLLAMA_HOST           — default "http://localhost:11434"
  OLLAMA_EXTRACT_MODEL  — default "qwen3.6:27b"

NOTE: change the model by setting OLLAMA_EXTRACT_MODEL or editing EXTRACT_MODEL
below. Pull it first, e.g.:  ollama pull qwen3.6:27b
"""

from __future__ import annotations

import os
import re

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EXTRACT_MODEL = os.getenv("OLLAMA_EXTRACT_MODEL", "qwen3.6:27b")

# A local 27B model can be slow per call; allow generous time.
_TIMEOUT = httpx.Timeout(600.0)

# Strip any stray reasoning block in case `think` is not honoured by the server.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def chat_json(system: str, user: str, model: str = EXTRACT_MODEL) -> str:
    """
    Send a system+user prompt to Ollama and return the model's text content,
    constrained to JSON. Raises httpx.HTTPError on transport/HTTP failure.
    """
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return _THINK_RE.sub("", content).strip()
