"""
parse_transcript.py

Reads a .txt transcript file exported from a Word document.

Expected format:
    Speaker Name  HH:MM
    Text content...

    Speaker Name  HH:MM
    More text...

Outputs transcript_episodes.json — a list of structured episode objects
ready to be consumed by ingest_transcript.py.
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRANSCRIPT_FILE = "The 26.44 Master Recording Session (EX REC 08)_transcript.docx.txt"
OUTPUT_FILE = "transcript_episodes.json"

# Base datetime for the session. The HH:MM timestamps in the transcript are
# treated as offsets from midnight on this date.
BASE_DATE = datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc)

# Matches a speaker header line: "Some Name  HH:MM" (one or more spaces between name and time)
SPEAKER_LINE_RE = re.compile(r"^(.+?)\s{2,}(\d{1,2}):(\d{2})\s*$")


def parse_transcript(path: Path) -> list[dict]:
    episodes = []
    current_speaker = None
    current_time = None
    current_lines: list[str] = []

    def flush():
        if current_speaker is None:
            return
        text = " ".join(current_lines).strip()
        if not text:
            return
        episodes.append({
            "speaker": current_speaker,
            "text": text,
            "timestamp": current_time,
        })

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            match = SPEAKER_LINE_RE.match(line)
            if match:
                flush()
                current_lines = []
                current_speaker = match.group(1).strip()
                hours = int(match.group(2))
                minutes = int(match.group(3))
                offset = timedelta(hours=hours, minutes=minutes)
                ts = BASE_DATE + offset
                current_time = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

    flush()
    return episodes


def main():
    path = Path(TRANSCRIPT_FILE)
    if not path.exists():
        print(f"Error: transcript file not found: {TRANSCRIPT_FILE}", file=sys.stderr)
        print("Place the transcript .txt file in the same directory as this script.", file=sys.stderr)
        sys.exit(1)

    episodes = parse_transcript(path)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

    speakers = sorted({ep["speaker"] for ep in episodes})
    total_chars = sum(len(ep["text"]) for ep in episodes)

    print(f"Parsed {len(episodes)} episodes")
    print(f"Speakers ({len(speakers)}): {', '.join(speakers)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
