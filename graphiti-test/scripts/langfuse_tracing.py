"""
langfuse_tracing.py

Optional Langfuse observability for this project. Tracing is on when both
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the environment
(typically via the project .env). Otherwise every observe/span call is a
no-op (LANGFUSE_TRACING_ENABLED=false) and the rest of the pipeline still
runs.

Defaults to a local Langfuse instance at http://localhost:3100 (see
docker-compose.langfuse.yml). No Langfuse Cloud account is required.

This module loads the project .env on import so later Langfuse client
construction sees the keys.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Local-first: the SDK otherwise defaults to Langfuse Cloud.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3100")

if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

from langfuse import (  # noqa: E402
    Evaluation,
    get_client,
    observe,
    propagate_attributes,
)


def tracing_configured() -> bool:
    """True when Langfuse API keys are present (traces will be exported)."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse():
    """Return the process-wide Langfuse client (no-op if tracing is disabled)."""
    return get_client()


def flush() -> None:
    """Flush buffered traces. Safe to call when tracing is disabled."""
    get_langfuse().flush()


__all__ = [
    "Evaluation",
    "flush",
    "get_langfuse",
    "observe",
    "propagate_attributes",
    "tracing_configured",
]
