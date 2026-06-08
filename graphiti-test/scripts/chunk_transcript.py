"""
chunk_transcript.py

Reads a .txt transcript file, parses it into individual speaker turns,
then groups those turns into chunks of approximately 3000 characters
without ever splitting mid-turn.

Usage:
  python scripts/chunk_transcript.py transcripts/<file>.txt

Output:
  output/transcript_chunks.json

Transcript format expected:
  Speaker Name  HH:MM
  Text content...

  Speaker Name  HH:MM
  More text...
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = _PROJECT_ROOT / "output" / "transcript_chunks.json"
TARGET_CHUNK_CHARS = 3000

# Handles both MM:SS and H:MM:SS timestamp formats.
SPEAKER_LINE_RE = re.compile(r"^(.+?)\s{2,}\d{1,2}(?::\d{2}){1,2}\s*$")

# Capture group version for preserving the raw header line.
SPEAKER_LINE_CAPTURE_RE = re.compile(
    r"^((.+?)\s{2,}(\d{1,2}(?::\d{2}){1,2}))\s*$"
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_turns(path: Path) -> list[dict]:
    """
    Parse the transcript into a list of turns.

    Each turn:
      {
        "speaker":   str,
        "timestamp": str,   # raw timestamp as it appears in the file
        "header":    str,   # the full original header line (e.g. "Name  04:16")
        "text":      str,   # body text, lines joined with spaces
      }
    """
    turns: list[dict] = []
    current_speaker: str | None = None
    current_timestamp: str | None = None
    current_header: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_speaker is None:
            return
        text = " ".join(current_lines).strip()
        if not text:
            return
        turns.append(
            {
                "speaker": current_speaker,
                "timestamp": current_timestamp,
                "header": current_header,
                "text": text,
            }
        )

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            m = SPEAKER_LINE_CAPTURE_RE.match(line)
            if m:
                flush()
                current_lines = []
                current_header = m.group(1)
                current_speaker = m.group(2).strip()
                current_timestamp = m.group(3)
            else:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

    flush()
    return turns

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def turn_body(turn: dict) -> str:
    """Reconstruct a turn as it should appear in the chunk body."""
    return f"{turn['header']}\n{turn['text']}\n"


def build_chunks(turns: list[dict], target: int = TARGET_CHUNK_CHARS) -> list[dict]:
    """
    Group turns into chunks of approximately `target` characters.

    Rules:
    - Never split mid-turn.
    - If adding the next turn would exceed `target`, close the current chunk.
    - A single turn longer than `target` gets its own chunk (never truncated).
    - Each chunk body starts at a speaker boundary.
    """
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_chars: int = 0
    current_turn_count: int = 0

    def flush_chunk() -> None:
        if not current_parts:
            return
        body = "\n".join(current_parts)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "char_count": len(body),
                "turn_count": current_turn_count,
                "body": body,
            }
        )

    for turn in turns:
        body = turn_body(turn)
        body_len = len(body)

        if current_parts and (current_chars + 1 + body_len) > target:
            # Adding this turn would exceed the target — close the current chunk.
            flush_chunk()
            current_parts = []
            current_chars = 0
            current_turn_count = 0

        if current_parts:
            # Blank line separator between turns inside the same chunk.
            current_parts.append("")
            current_chars += 1  # the "\n" the blank line adds during join

        current_parts.append(body.rstrip("\n"))
        current_chars += body_len
        current_turn_count += 1

    flush_chunk()
    return chunks

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python chunk_transcript.py <transcript.txt>",
            file=sys.stderr,
        )
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading transcript: {path}")

    turns = parse_turns(path)
    if not turns:
        print("Warning: no speaker turns found in transcript.", file=sys.stderr)
        sys.exit(0)

    chunks = build_chunks(turns)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Summary statistics
    char_counts = [c["char_count"] for c in chunks]
    total_chars = sum(char_counts)
    avg_chars = total_chars / len(chunks) if chunks else 0
    min_chars = min(char_counts) if char_counts else 0
    max_chars = max(char_counts) if char_counts else 0
    min_idx = char_counts.index(min_chars)
    max_idx = char_counts.index(max_chars)

    print()
    print(f"Total turns parsed : {len(turns)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size  : {avg_chars:,.0f} chars")
    print(f"Smallest chunk      : {min_chars:,} chars  (chunk {min_idx})")
    print(f"Largest chunk       : {max_chars:,} chars  (chunk {max_idx})")
    print()
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
