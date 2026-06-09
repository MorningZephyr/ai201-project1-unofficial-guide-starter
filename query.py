"""Milestone 5 — Grounded generation.

Wires retrieval to Groq's llama-3.3-70b-versatile. The system prompt *enforces*
grounding (answer only from the retrieved context, otherwise decline), and source
attribution is added programmatically from retrieval metadata rather than being
left to the model to invent.

    from query import ask
    result = ask("How hard are Professor Shostak's exams?")
    print(result["answer"], result["sources"])
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from retrieve import RetrievedChunk, retrieve

load_dotenv(Path(__file__).parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# Cosine distance above which we consider even the *best* match too weak to be
# real coverage. Tuned from observed in-domain distances (~0.2-0.45) vs.
# out-of-domain (~0.9+). See planning.md > Anticipated Challenges.
RELEVANCE_CUTOFF = 0.85

NOT_ENOUGH_INFO = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Computer Science "
    "professors at Hunter College using ONLY the student reviews provided in the "
    "CONTEXT section. Follow these rules strictly:\n"
    "1. Use only facts stated in the CONTEXT. Do not use any outside or prior "
    "knowledge about these professors or courses.\n"
    "2. If the CONTEXT does not contain enough information to answer, reply with "
    f'exactly: "{NOT_ENOUGH_INFO}"\n'
    "3. When reviews disagree, reflect that (e.g. note that opinions are mixed) "
    "instead of picking one side.\n"
    "4. Do not invent professor names, courses, grades, or policies that are not "
    "in the CONTEXT.\n"
    "5. Be concise: 2-5 sentences."
)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c.source}, professor: {c.professor})\n{c.text}")
    return "\n\n".join(blocks)


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Programmatic source attribution: dedupe by file, preserve retrieval order."""
    seen, sources = set(), []
    for c in chunks:
        label = f"{c.professor} ({c.source})"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def ask(question: str, k: int = TOP_K) -> dict:
    """Retrieve, generate a grounded answer, and attach sources.

    Returns {"answer": str, "sources": list[str], "chunks": list[RetrievedChunk]}.
    """
    chunks = retrieve(question, k=k)

    # Out-of-coverage guard: if nothing is even moderately close, don't bother the
    # LLM — decline up front so we can't hallucinate from training knowledge.
    if not chunks or chunks[0].distance > RELEVANCE_CUTOFF:
        return {"answer": NOT_ENOUGH_INFO, "sources": [], "chunks": chunks}

    user_prompt = (
        f"CONTEXT:\n{_format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above."
    )

    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content.strip()

    # If the model declined, don't attach sources (nothing was actually used).
    sources = [] if answer.strip() == NOT_ENOUGH_INFO else _unique_sources(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks}


def main() -> None:
    demo_questions = [
        "What do students say about the exams in Professor Pavel Shostak's classes?",
        "What is unusual about Professor Sven Dietrich's exams and quizzes?",
        "What is the best pizza place near campus?",  # out-of-coverage
    ]
    for q in demo_questions:
        result = ask(q)
        print("=" * 78)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print()


if __name__ == "__main__":
    main()
