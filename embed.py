"""Milestone 4 — Embed chunks and load them into ChromaDB.

Loads the chunks produced by ingest.py, embeds them locally with
all-MiniLM-L6-v2 (sentence-transformers), and upserts them into a persistent
ChromaDB collection with metadata (source, professor, course, chunk index).

Run directly to (re)build the vector store:
    python embed.py
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHUNKS_PATH, build_chunks, save_chunks

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "rmp_reviews"

# Cosine space: distances range 0 (identical) .. 2 (opposite). We treat < 0.5 as
# a strong match (see planning.md > Retrieval Approach).
_DISTANCE_SPACE = "cosine"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load (and cache) the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(create: bool = False):
    """Return the Chroma collection, optionally (re)creating it fresh."""
    client = get_client()
    if create:
        # Drop any stale collection so re-runs don't duplicate or mix old data.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": _DISTANCE_SPACE}
        )
    return client.get_collection(COLLECTION_NAME)


def build_index() -> int:
    """Embed all chunks and store them in ChromaDB. Returns the chunk count."""
    chunks = build_chunks()
    save_chunks(chunks)  # keep chunks.json in sync
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL_NAME} ...")

    model = get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    collection = get_collection(create=True)
    collection.add(
        ids=[f"{c.source}#{c.chunk_index}" for c in chunks],
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[
            {
                "source": c.source,
                "professor": c.professor,
                "course": c.course,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB at {CHROMA_DIR}")
    return len(chunks)


def main() -> None:
    if not CHUNKS_PATH.exists():
        print("chunks.json not found — building from documents/ first.")
    count = build_index()
    print(f"Done. Vector store ready with {count} chunks.")


if __name__ == "__main__":
    main()
