"""Milestone 6 — Run the evaluation plan.

Runs the 5 test questions from planning.md (plus a couple of probing questions
used to find a failure case) through the full pipeline and prints, for each:
the answer, the cited sources, and the retrieval distances. Results are recorded
by hand in README.md.

    python evaluate.py
"""

from __future__ import annotations

from query import ask

EVAL_QUESTIONS = [
    "What do students say about the exams in Professor Pavel Shostak's classes?",
    "How heavy is the workload and difficulty in Professor Saad Mneimneh's CSCI150 class?",
    "Why do students find Professor Tong Yi's lectures hard to follow?",
    "What do reviews say about Professor Melissa Lynch's punctuality and responsiveness to emails?",
    "What is unusual about Professor Sven Dietrich's exams and quizzes?",
]

PROBING_QUESTIONS = [
    "Who is the easiest Computer Science professor to take at Hunter?",
    "How accessible is Professor Saptarshi Debroy during office hours?",
]


def run(question: str) -> None:
    result = ask(question)
    print("=" * 80)
    print(f"Q: {question}")
    print("-" * 80)
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print("Retrieval (distance | professor | course):")
    for r in result["chunks"]:
        print(f"   {r.distance:.3f} | {r.professor} | {r.course}")
    print()


def main() -> None:
    print("\n########## EVALUATION QUESTIONS (planning.md) ##########\n")
    for q in EVAL_QUESTIONS:
        run(q)
    print("\n########## PROBING QUESTIONS (failure-case hunting) ##########\n")
    for q in PROBING_QUESTIONS:
        run(q)


if __name__ == "__main__":
    main()
