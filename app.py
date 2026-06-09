"""Milestone 5 — Gradio web interface.

A minimal UI over the grounded RAG pipeline. Enter a question about a Hunter
College CS professor; the app retrieves relevant student reviews, generates a
grounded answer, and shows which source documents it drew from.

    python app.py
    # then open http://localhost:7860
"""

from __future__ import annotations

import gradio as gr

from query import ask

EXAMPLES = [
    "What do students say about the exams in Professor Pavel Shostak's classes?",
    "How heavy is the workload in Professor Saad Mneimneh's CSCI150 class?",
    "Why do students find Professor Tong Yi's lectures hard to follow?",
    "What is unusual about Professor Sven Dietrich's exams and quizzes?",
    "Is Professor Raffi Khatchadourian an easy grader?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    sources = result["sources"]
    sources_text = "\n".join(f"• {s}" for s in sources) if sources else "(no sources cited)"
    return result["answer"], sources_text


with gr.Blocks(title="The Unofficial Guide — Hunter College CS Professors") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask about **Computer Science professors at Hunter College**. Answers are "
        "grounded only in student reviews collected from RateMyProfessors — if the "
        "reviews don't cover something, the system will say so."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. How hard are Professor Shostak's exams?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
