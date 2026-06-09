"""Milestone 4 — Retrieval.

Embeds a query with the same model used for indexing and returns the top-k most
similar chunks from ChromaDB, each with its source metadata and cosine distance.

Run directly to test retrieval against sample evaluation queries:
    python retrieve.py
"""

from __future__ import annotations

from dataclasses import dataclass

from embed import get_collection, get_model

DEFAULT_K = 5


@dataclass
class RetrievedChunk:
    text: str
    source: str
    professor: str
    course: str
    distance: float


def retrieve(query: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
    """Return the top-k most relevant chunks for a query."""
    model = get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    return [
        RetrievedChunk(
            text=doc,
            source=meta.get("source", "unknown"),
            professor=meta.get("professor", "unknown"),
            course=meta.get("course", "N/A"),
            distance=dist,
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def main() -> None:
    sample_queries = [
        "What do students say about the exams in Professor Pavel Shostak's classes?",
        "How heavy is the workload and difficulty in Professor Saad Mneimneh's CSCI150 class?",
        "Why do students find Professor Tong Yi's lectures hard to follow?",
    ]
    for q in sample_queries:
        print("=" * 78)
        print(f"QUERY: {q}")
        print("=" * 78)
        for i, r in enumerate(retrieve(q), 1):
            print(f"  [{i}] distance={r.distance:.3f} | {r.professor} | "
                  f"{r.course} | {r.source}")
            review_line = r.text.split("\n")[-1]
            print(f"      {review_line[:160]}")
        print()


if __name__ == "__main__":
    main()
