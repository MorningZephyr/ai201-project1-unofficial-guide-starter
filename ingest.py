"""Milestone 3 — Document ingestion and chunking.

Loads the cleaned RateMyProfessors review files from documents/, cleans residual
noise, and splits each document into one chunk per review (review-aware chunking,
per planning.md). Each chunk is self-contained: it carries the professor name and
course so it is retrievable on its own.

Run directly to inspect the output:
    python ingest.py
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "documents"
CHUNKS_PATH = Path(__file__).parent / "chunks.json"

# Review-aware chunking parameters (see planning.md > Chunking Strategy).
MAX_CHUNK_TOKENS = 280   # cap; oversized reviews are sub-split
SUBSPLIT_TOKENS = 250    # target size when a single review exceeds the cap
SUBSPLIT_OVERLAP = 30    # word overlap only used inside an oversized review


@dataclass
class Chunk:
    text: str
    source: str       # filename, e.g. rmp_shostak.txt
    professor: str
    course: str
    chunk_index: int  # position of this chunk within its source document


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~0.75 words per token) — good enough to enforce a cap."""
    words = len(text.split())
    return max(1, round(words / 0.75))


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    """Load every .txt file from disk as (filename, raw_text)."""
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")
    return docs


def clean_text(text: str) -> str:
    """Remove residual noise that doesn't belong to the review content.

    The documents were already cleaned of site boilerplate when collected, but we
    defensively strip any stray HTML tags, decode HTML entities (&amp;, &#39;),
    drop leftover RMP UI lines, and normalize whitespace.
    """
    text = html.unescape(text)               # &amp; -> &, &#39; -> '
    text = re.sub(r"<[^>]+>", "", text)       # any stray HTML tags
    noise = re.compile(
        r"^\s*(Helpful|Load More Ratings|CA Notice.*|Do Not Sell.*|©.*|"
        r"Similar Professors|I'm Professor.*)\s*$",
        re.IGNORECASE,
    )
    lines = [ln.rstrip() for ln in text.splitlines() if not noise.match(ln)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse big blank gaps
    return text.strip()


def _parse_header(header: str) -> str:
    m = re.search(r"^Professor:\s*(.+)$", header, re.MULTILINE)
    return m.group(1).strip() if m else "Unknown"


def _parse_course(meta_line: str) -> str:
    m = re.search(r"Course:\s*([^|]+)", meta_line)
    return m.group(1).strip() if m else "N/A"


def _word_window_split(text: str, size: int, overlap: int) -> list[str]:
    """Fallback splitter for an oversized single review (rarely needed)."""
    words = text.split()
    if not words:
        return []
    pieces, start = [], 0
    while start < len(words):
        end = start + size
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return pieces


def chunk_document(filename: str, raw_text: str) -> list[Chunk]:
    """Split one cleaned document into review-level chunks."""
    text = clean_text(raw_text)

    # Split into [header, review-1, review-2, ...] on "Review N" markers.
    parts = re.split(r"^Review\s+\d+\s*$", text, flags=re.MULTILINE)
    header = parts[0]
    review_blocks = [p.strip() for p in parts[1:] if p.strip()]
    professor = _parse_header(header)

    chunks: list[Chunk] = []
    for block in review_blocks:
        block_lines = block.splitlines()
        meta_line = block_lines[0] if block_lines else ""
        course = _parse_course(meta_line)

        # Self-contained chunk: professor + the review's metadata + the review text.
        chunk_body = f"Professor {professor} review.\n{block}"

        if estimate_tokens(chunk_body) <= MAX_CHUNK_TOKENS:
            pieces = [chunk_body]
        else:
            review_text = "\n".join(block_lines[1:])
            prefix = f"Professor {professor} ({course}) review."
            pieces = [
                f"{prefix}\n{piece}"
                for piece in _word_window_split(review_text, SUBSPLIT_TOKENS, SUBSPLIT_OVERLAP)
            ]

        for piece in pieces:
            piece = piece.strip()
            if len(piece) == 0:          # filter empty chunks
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source=filename,
                    professor=professor,
                    course=course,
                    chunk_index=len(chunks),
                )
            )
    return chunks


def build_chunks() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for filename, raw_text in load_documents():
        all_chunks.extend(chunk_document(filename, raw_text))
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}\n")

    # Show one cleaned document so we can verify cleaning worked.
    sample_name, sample_raw = docs[0]
    print("=" * 70)
    print(f"CLEANED SAMPLE DOCUMENT: {sample_name}")
    print("=" * 70)
    print(clean_text(sample_raw)[:600])
    print("...\n")

    chunks = build_chunks()
    save_chunks(chunks)

    print("=" * 70)
    print(f"TOTAL CHUNKS: {len(chunks)}  (saved to {CHUNKS_PATH.name})")
    print("=" * 70)

    lengths = [estimate_tokens(c.text) for c in chunks]
    print(f"Token estimate per chunk — min {min(lengths)}, "
          f"max {max(lengths)}, avg {sum(lengths) // len(lengths)}\n")

    # Print 5 representative chunks spread across the corpus for inspection.
    step = max(1, len(chunks) // 5)
    sample_indices = list(range(0, len(chunks), step))[:5]
    print("5 REPRESENTATIVE CHUNKS:")
    for i in sample_indices:
        c = chunks[i]
        print("-" * 70)
        print(f"[chunk {i}] source={c.source} | professor={c.professor} | "
              f"course={c.course} | ~{estimate_tokens(c.text)} tokens")
        print(c.text)
    print("-" * 70)


if __name__ == "__main__":
    main()
