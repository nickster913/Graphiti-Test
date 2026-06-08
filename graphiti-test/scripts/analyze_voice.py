"""
analyze_voice.py

Reads a transcript .txt file, extracts only RayJai Babauta's speaking turns,
and produces two analysis files:

  output/rayjai_word_freq.txt    — top words, bigrams, and trigrams
  output/rayjai_word_context.txt — example sentences for his top 20 non-generic words

Usage:
  python scripts/analyze_voice.py                        # auto-detects .txt in transcripts/
  python scripts/analyze_voice.py transcripts/<file>.txt # explicit file

Transcript format expected:
  Speaker Name  HH:MM
  Text content...

  Speaker Name  HH:MM
  More text...
"""

import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

TARGET_SPEAKER_RE = re.compile(r"rayjai\s+babauta", re.IGNORECASE)

# Handles both MM:SS and H:MM:SS timestamp formats in speaker headers
SPEAKER_LINE_RE = re.compile(r"^(.+?)\s{2,}\d{1,2}(?::\d{2}){1,2}\s*$")

STOPWORDS = {
    # Basic function words (as specified)
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "of", "to", "in", "that", "it", "you", "i", "we",
    "so", "just", "like", "he", "she", "they", "them", "their",
    "this", "what", "with", "have", "had", "has", "do", "did", "does",
    "for", "on", "at", "by", "from", "as", "up", "not", "my", "your",
    "our", "me", "him", "her", "its", "if", "all", "can", "will", "one",
    "get", "got", "about", "which", "there", "here", "when", "how",
    "would", "could", "should", "into", "out", "over", "than", "then",
    "some", "no", "yeah", "yes", "okay", "oh", "um", "uh", "right",
    "know", "think", "mean", "because", "very", "really", "actually",
    "gonna", "gotta", "want", "need", "see", "say", "said", "go", "going",
    "come", "way", "thing", "things", "time", "people", "make", "kind",
    "lot", "even", "much", "also", "back", "still", "any", "more",
    "well", "now", "too", "who", "those", "these",
    # Common contractions (surface as single tokens after apostrophe-aware tokenization)
    "it's", "that's", "i'm", "they're", "you're", "there's", "don't",
    "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "i've",
    "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd",
    "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll",
    "they'll", "doesn't", "didn't", "wouldn't", "couldn't", "shouldn't",
    "haven't", "hasn't", "hadn't", "let's", "that'll", "who's", "what's",
    "where's", "when's", "how's", "here's", "it'll", "it'd",
    # Filler / discourse words
    "blah", "boo",
}

FREQ_OUTPUT = _PROJECT_ROOT / "output" / "rayjai_word_freq.txt"
CONTEXT_OUTPUT = _PROJECT_ROOT / "output" / "rayjai_word_context.txt"

TOP_WORDS = 50
TOP_BIGRAMS = 30
TOP_TRIGRAMS = 20
TOP_CONTEXT_WORDS = 20
CONTEXT_EXAMPLES = 3

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_transcript(path: Path) -> list[dict]:
    """Return a list of {speaker, text} dicts for every turn in the transcript."""
    episodes = []
    current_speaker = None
    current_lines: list[str] = []

    def flush():
        if current_speaker is None:
            return
        text = " ".join(current_lines).strip()
        if text:
            episodes.append({"speaker": current_speaker, "text": text})

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            match = SPEAKER_LINE_RE.match(line)
            if match:
                flush()
                current_lines = []
                current_speaker = match.group(1).strip()
            else:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

    flush()
    return episodes


def extract_target_turns(episodes: list[dict]) -> list[str]:
    """Return only the text turns belonging to the target speaker."""
    return [
        ep["text"]
        for ep in episodes
        if TARGET_SPEAKER_RE.search(ep["speaker"])
    ]

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

TOKENIZE_RE = re.compile(r"[a-z']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens (apostrophes kept for contractions)."""
    return TOKENIZE_RE.findall(text.lower())


def sentences_from_turns(turns: list[str]) -> list[str]:
    """Split each speaking turn into individual sentences."""
    sentences = []
    for turn in turns:
        for sent in SENTENCE_SPLIT_RE.split(turn):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def is_content_word(word: str) -> bool:
    return word not in STOPWORDS and len(word) > 2


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def is_content_ngram(gram: tuple) -> bool:
    """Keep only ngrams where at least the first and last token are content words."""
    return is_content_word(gram[0]) and is_content_word(gram[-1])

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def word_frequency(turns: list[str]) -> tuple[Counter, Counter, Counter]:
    """Return (word_counts, bigram_counts, trigram_counts) filtered for content."""
    all_tokens: list[str] = []
    for turn in turns:
        all_tokens.extend(tokenize(turn))

    word_counts = Counter(w for w in all_tokens if is_content_word(w))
    bigram_counts = Counter(g for g in ngrams(all_tokens, 2) if is_content_ngram(g))
    trigram_counts = Counter(g for g in ngrams(all_tokens, 3) if is_content_ngram(g))

    return word_counts, bigram_counts, trigram_counts


def context_examples(
    top_words: list[str],
    sentences: list[str],
    n_examples: int = CONTEXT_EXAMPLES,
) -> dict[str, list[str]]:
    """For each word, collect up to n_examples sentences that contain it."""
    examples: dict[str, list[str]] = {w: [] for w in top_words}
    for sent in sentences:
        tokens = set(tokenize(sent))
        for word in top_words:
            if word in tokens and len(examples[word]) < n_examples:
                examples[word].append(sent)
    return examples

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt_gram(gram: tuple) -> str:
    return " ".join(gram)


def write_freq_file(
    word_counts: Counter,
    bigram_counts: Counter,
    trigram_counts: Counter,
    out_path: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RAYJAI BABAUTA — WORD FREQUENCY ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"TOP {TOP_WORDS} SINGLE WORDS\n")
        f.write("-" * 40 + "\n")
        for rank, (word, count) in enumerate(word_counts.most_common(TOP_WORDS), 1):
            f.write(f"  {rank:>3}. {word:<30} {count}\n")

        f.write(f"\nTOP {TOP_BIGRAMS} BIGRAMS (2-word phrases)\n")
        f.write("-" * 40 + "\n")
        for rank, (gram, count) in enumerate(bigram_counts.most_common(TOP_BIGRAMS), 1):
            f.write(f"  {rank:>3}. {fmt_gram(gram):<30} {count}\n")

        f.write(f"\nTOP {TOP_TRIGRAMS} TRIGRAMS (3-word phrases)\n")
        f.write("-" * 40 + "\n")
        for rank, (gram, count) in enumerate(trigram_counts.most_common(TOP_TRIGRAMS), 1):
            f.write(f"  {rank:>3}. {fmt_gram(gram):<35} {count}\n")


def write_context_file(
    examples: dict[str, list[str]],
    word_counts: Counter,
    out_path: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RAYJAI BABAUTA — WORD CONTEXT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            f"Top {TOP_CONTEXT_WORDS} non-generic words with up to "
            f"{CONTEXT_EXAMPLES} example sentences each.\n\n"
        )
        for word, sents in examples.items():
            count = word_counts[word]
            f.write(f"{'─' * 50}\n")
            f.write(f'"{word}"  (used {count}x)\n')
            f.write(f"{'─' * 50}\n")
            if sents:
                for i, sent in enumerate(sents, 1):
                    f.write(f"  {i}. {sent}\n")
            else:
                f.write("  (no sentence-level examples found)\n")
            f.write("\n")

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def find_transcript() -> Path:
    """Auto-detect a .txt file in the transcripts/ folder."""
    candidates = sorted((_PROJECT_ROOT / "transcripts").glob("*.txt"))
    if not candidates:
        print("Error: no .txt transcript file found in transcripts/.", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print(f"Multiple .txt files found; using: {candidates[0]}")
    return candidates[0]


def main() -> None:
    if len(sys.argv) > 1:
        transcript_path = Path(sys.argv[1])
        if not transcript_path.exists():
            print(f"Error: file not found: {transcript_path}", file=sys.stderr)
            sys.exit(1)
    else:
        transcript_path = find_transcript()
        # Also check the root-level transcript as a fallback if transcripts/ is empty
        if not transcript_path.exists():
            print(f"Error: file not found: {transcript_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Reading transcript: {transcript_path}")

    episodes = parse_transcript(transcript_path)
    turns = extract_target_turns(episodes)

    if not turns:
        print("Warning: no turns found for RayJai Babauta. Check the speaker name in the transcript.")
        sys.exit(0)

    print(f"Found {len(turns)} speaking turn(s) from RayJai Babauta.")

    # --- Frequency analysis ---
    word_counts, bigram_counts, trigram_counts = word_frequency(turns)

    # --- Context analysis ---
    sentences = sentences_from_turns(turns)
    top_content_words = [w for w, _ in word_counts.most_common(TOP_CONTEXT_WORDS)]
    examples = context_examples(top_content_words, sentences)

    # --- Write output files ---
    write_freq_file(word_counts, bigram_counts, trigram_counts, FREQ_OUTPUT)
    print(f"Word frequency written to: {FREQ_OUTPUT}")

    write_context_file(examples, word_counts, CONTEXT_OUTPUT)
    print(f"Word context written to:   {CONTEXT_OUTPUT}")

    # --- Terminal summary ---
    print()
    print("=" * 60)
    print("TOP 20 WORDS")
    print("=" * 60)
    for rank, (word, count) in enumerate(word_counts.most_common(20), 1):
        bar = "█" * min(count, 40)
        print(f"  {rank:>2}. {word:<25} {count:>4}  {bar}")

    print()
    print("=" * 60)
    print("TOP 10 PHRASES (bigrams + trigrams combined)")
    print("=" * 60)
    all_phrases = [
        (fmt_gram(g), c) for g, c in bigram_counts.most_common(20)
    ] + [
        (fmt_gram(g), c) for g, c in trigram_counts.most_common(10)
    ]
    all_phrases.sort(key=lambda x: x[1], reverse=True)
    for rank, (phrase, count) in enumerate(all_phrases[:10], 1):
        bar = "█" * min(count, 40)
        print(f"  {rank:>2}. {phrase:<35} {count:>4}  {bar}")

    print()
    print(f"Total unique content words: {len(word_counts)}")
    print(f"Total sentences analyzed:  {len(sentences)}")


if __name__ == "__main__":
    main()
