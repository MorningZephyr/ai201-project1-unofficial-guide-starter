# The Unofficial Guide — Hunter College CS Professors

A retrieval-augmented question-answering system over student reviews of Computer
Science professors at Hunter College (CUNY). Ask a natural-language question
(e.g. "How hard are Professor Shostak's exams?") and get an answer grounded only
in the collected reviews, with the source document(s) cited.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# Add your Groq key (free at https://console.groq.com)
copy .env.example .env          # then edit .env and set GROQ_API_KEY

python embed.py                 # build the ChromaDB vector store from documents/
python app.py                   # launch the Gradio UI at http://localhost:7860
```

Pipeline scripts: `ingest.py` (load + chunk) → `embed.py` (embed + store) →
`retrieve.py` (search) → `query.py` (grounded generation) → `app.py` (UI).
`evaluate.py` runs the evaluation questions.

---

## Domain

This system covers **student reviews of Computer Science professors at Hunter
College (CUNY)**, collected from RateMyProfessors. This knowledge is valuable
because the official CUNY course catalog only lists course titles and
prerequisites — it says nothing about how a professor actually teaches, how
exams are structured and curved, how heavy the workload is, or whether the
professor is responsive and accessible. That experiential knowledge is what
students most want before registering, and it is hard to find through official
channels: it lives in scattered, JavaScript-rendered review pages that can't be
searched or compared in aggregate. Grounding answers in the actual review text
lets a student ask a specific question and get a sourced, balanced summary across
multiple opinions instead of reading dozens of individual reviews.

---

## Document Sources

Ten RateMyProfessors pages for Hunter College CS professors. Each page was
fetched, stripped of site boilerplate (navigation, footers, "Helpful" counters,
ads, "Load More Ratings"), and saved as a plain `.txt` file in `documents/`. The
professors span the full ratings range (1.2 to 4.3 / 5) and a variety of courses
(intro programming, data structures, operating systems, machine learning,
security), giving the corpus diverse perspectives.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | RateMyProfessors — Pavel Shostak | RMP reviews (.txt) | `documents/rmp_shostak.txt` — https://www.ratemyprofessors.com/professor/1823870 |
| 2 | RateMyProfessors — Saad Mneimneh | RMP reviews (.txt) | `documents/rmp_mneimneh.txt` — https://www.ratemyprofessors.com/professor/926045 |
| 3 | RateMyProfessors — Tong Yi | RMP reviews (.txt) | `documents/rmp_yi.txt` — https://www.ratemyprofessors.com/professor/2634841 |
| 4 | RateMyProfessors — Melissa Lynch | RMP reviews (.txt) | `documents/rmp_lynch.txt` — https://www.ratemyprofessors.com/professor/2505090 |
| 5 | RateMyProfessors — Saptarshi Debroy | RMP reviews (.txt) | `documents/rmp_debroy.txt` — https://www.ratemyprofessors.com/professor/2220139 |
| 6 | RateMyProfessors — Sven Dietrich | RMP reviews (.txt) | `documents/rmp_dietrich.txt` — https://www.ratemyprofessors.com/professor/2674099 |
| 7 | RateMyProfessors — Susan Epstein | RMP reviews (.txt) | `documents/rmp_epstein.txt` — https://www.ratemyprofessors.com/professor/192300 |
| 8 | RateMyProfessors — Raffi Khatchadourian | RMP reviews (.txt) | `documents/rmp_khatchadourian.txt` — https://www.ratemyprofessors.com/professor/2259095 |
| 9 | RateMyProfessors — Raj Korpan | RMP reviews (.txt) | `documents/rmp_korpan.txt` — https://www.ratemyprofessors.com/professor/2659561 |
| 10 | RateMyProfessors — Mohammad Manshaei | RMP reviews (.txt) | `documents/rmp_manshaei.txt` — https://www.ratemyprofessors.com/professor/3066287 |

---

## Chunking Strategy

**Chunk size:** one student review per chunk, capped at ~280 tokens. Observed
chunk sizes ranged from 44 to 149 tokens (avg ~102). Each chunk is prefixed with
a context line — professor name plus the review's course and quality/difficulty
ratings — so it stands on its own.

**Overlap:** none between separate reviews. Reviews are independent opinions, so
overlapping them would only duplicate unrelated content. (A 30-token overlap is
applied only in the rare case that a single review exceeds the 280-token cap and
must be sub-split; with this corpus that never triggered.)

**Why these choices fit my documents:** Each RateMyProfessors review is a short,
self-contained opinion (~40–90 tokens). Splitting on the natural review boundary
keeps every chunk a single "complete thought" (e.g. "Shostak's exams are 90% of
the grade and require memorizing exact definitions"), which is exactly what
semantic search needs to match a specific question. A naive fixed 200–300 token
window would merge two or three *unrelated* student opinions into one chunk and
dilute the embedding. Prefixing the professor name makes name-based queries
retrievable even when the review text never repeats the name — this is why every
top result in retrieval testing came from the correct professor.

**Preprocessing before chunking:** decode HTML entities, strip any stray HTML
tags, remove leftover RMP UI lines ("Helpful", "Load More Ratings", footers,
"I'm Professor X"), and normalize whitespace (`clean_text()` in `ingest.py`).

**Final chunk count:** 48 chunks across the 10 documents.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. It runs locally
with no API key or rate limit, produces 384-dimensional embeddings, and embeds
the entire 48-chunk corpus in seconds. Embeddings are L2-normalized and stored in
a persistent **ChromaDB** collection using cosine distance, with per-chunk
metadata (source filename, professor, course, chunk index).

**Production tradeoff reflection:** If I were deploying this for real students
and cost weren't a constraint, I'd weigh: **(1) accuracy on domain-specific
text** — a larger model such as OpenAI `text-embedding-3-large` (or a model
fine-tuned on reviews) would better distinguish near-synonyms like "curved" vs.
"lenient" or "tough grader" vs. "picky"; **(2) context length** — MiniLM
truncates around 256 tokens, which is fine for short reviews but would lose
information if I later ingested long syllabi or thread discussions; **(3)
multilingual support** — several reviews mention non-native-English students, so
a multilingual model would help if reviews appeared in other languages; **(4)
latency vs. control** — MiniLM is local and instant, while a hosted embedding API
adds network latency and an external dependency but scales better and offloads
memory. For this small, short-text corpus the accuracy gains wouldn't justify the
added cost and latency, so MiniLM is the right default.

---

## Grounded Generation

**System prompt grounding instruction:** The Groq `llama-3.3-70b-versatile` model
is given a system prompt that *enforces* (not merely suggests) grounding. The key
instructions are: "answer questions ... using ONLY the student reviews provided
in the CONTEXT section"; "Use only facts stated in the CONTEXT. Do not use any
outside or prior knowledge about these professors or courses"; and "If the
CONTEXT does not contain enough information to answer, reply with exactly: *I
don't have enough information on that.*" The model is also told to reflect
disagreement when reviews conflict and never to invent professors, courses, or
grades. Temperature is set low (0.2) to reduce drift from the context.

**Structural grounding choices:** (1) An **out-of-coverage guard** in `query.py`
short-circuits before the LLM is ever called: if the best retrieved chunk has a
cosine distance above 0.85, the system returns the "not enough information"
message directly, so it cannot fall back on training knowledge for topics the
corpus doesn't cover. (2) The retrieved chunks are passed as an explicitly
delimited, numbered `CONTEXT` block, each labeled with its source and professor.

**How source attribution is surfaced:** Sources are attached
**programmatically**, not by the LLM. After retrieval, `query.py` dedupes the
retrieved chunks' metadata into a source list (`Professor Name (filename)`) and
returns it alongside the answer; the Gradio UI shows it in a separate "Retrieved
from" panel. If the model declines (returns the "not enough information"
message), no sources are attached because none were actually used.

---

## Evaluation Report

Run with `python evaluate.py`. All five questions are from `planning.md`.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Professor Pavel Shostak's exams? | Exams ~90% of grade, strict, require memorizing his exact definitions/terms; opinions split. | Said exams are "test heavy" and strict, looks for specific words/terms, ~90% of grade; noted mixed opinions (manageable if you attend vs. "learn nothing"). | Relevant (top distances 0.23–0.31, all from Shostak) | **Accurate** |
| 2 | How heavy is the workload/difficulty in Professor Saad Mneimneh's CSCI150 class? | Hard class, heavy workload (1–2 weeks, 3–8 hrs/day for exams), tough/ungraded HW, but curved and he's accessible; difficulty attributed to material. | Reported high workload, 3–8 hrs/day for 1–2 weeks, "test heavy"; noted mixed views on whether difficulty is the material or teaching. | Relevant (0.33–0.42, all Mneimneh) | **Accurate** |
| 3 | Why do students find Professor Tong Yi's lectures hard to follow? | Her accent/English makes lectures hard to understand; students rely on tutoring, recitations, posted past exams. | Attributed difficulty to her accent being hard to understand, quoting "lectures were pretty useless" and "Can't understand her." | Relevant (0.26–0.40, all Yi) | **Accurate** |
| 4 | What do reviews say about Professor Melissa Lynch's punctuality and email responsiveness? | Often ~30 min late, slow to return grades (~2 weeks), doesn't respond to emails; minority still positive. | Said she shows up ~30 min late and does not respond to emails; noted overall opinions are mixed. | Relevant (0.37–0.43, all Lynch) | **Accurate** |
| 5 | What is unusual about Professor Sven Dietrich's exams and quizzes? | Pen-only, assigned seats, individually printed exams; pop quizzes on random/movie content; multi-answer MC with no partial credit; good curve. | Listed individual per-student exams, pen only, niche movie questions, multi-answer quizzes with no partial credit, assigned seating. | Relevant (0.18–0.37, all Dietrich) | **Accurate** |

**Summary:** 5/5 accurate with relevant, on-professor retrieval. Each top result
had a cosine distance well below 0.5, and every cited source was the correct
professor — a direct payoff of prefixing chunks with the professor name.

---

## Failure Case Analysis

**Question that failed:** "Who is the easiest Computer Science professor to take
at Hunter?" (a comparative/superlative probing question, run via `evaluate.py`).

**What the system returned:** A muddled, partly incoherent answer naming Raj
Korpan and Mohammad Manshaei as "easier," with a broken clause ("a difficulty
rating of 2.0 is not present, but 3.0 is, however ..."). Its **top retrieved
chunk was actually Susan Epstein** (distance 0.450) — one of the *hardest*,
toughest-grading professors in the corpus (difficulty 4.4/5) — so the retrieval
neighborhood was off-target, and the answer never actually surveyed all ten
professors.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval +
chunk-schema** failure, not a generation failure. The system answers from the
top-k = 5 nearest chunks, which is a *local* neighborhood of the embedding space,
not a global view of the corpus. A superlative question ("easiest") requires
ranking all professors by difficulty, but (a) retrieval only ever surfaces 5
chunks, so 8–9 professors are never even seen by the LLM, and (b) difficulty is
stored only as free text inside each chunk, not as a structured numeric field the
system can sort on. Worse, the phrase "easiest professor" embeds near reviews
that *discuss* difficulty in general (including Epstein's "the workload is heavy
but worth it"), so semantic similarity actively pulls in a hard professor. The
distances here (0.45–0.54, straddling the 0.5 quality line) are also visibly
weaker than the 0.18–0.43 seen on the five answerable questions, which is the
warning sign.

**What I would change to fix it:** (1) Store the numeric quality/difficulty and
"would take again" values as ChromaDB metadata fields and add a separate
aggregation path for comparative/superlative questions that ranks professors
directly instead of relying on semantic retrieval; (2) detect superlative
questions and either raise top-k to cover all professors or run one retrieval per
professor; (3) tighten the relevance guard for these queries so a 0.45+ scattered
neighborhood triggers a "this needs a comparison I can't reliably make from the
reviews" response rather than a confident guess.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking
Strategy and Retrieval Approach sections *before* coding forced me to reason
about the structure of my documents up front. Because I had already decided that
each chunk should be a single self-contained review prefixed with the professor's
name, the embedding and retrieval stages "just worked" on the first try — every
test query returned chunks from the correct professor with distances under 0.5.
The planning diagram also kept the five stages cleanly separated into
`ingest.py` / `embed.py` / `retrieve.py` / `query.py` / `app.py`, so each piece
was easy to test in isolation before wiring the next one in.

**One way your implementation diverged from the spec, and why:** My original
spec said chunk at a fixed 200–300 tokens with 30-token overlap. Once I loaded
the documents I realized each review is only ~40–90 tokens, so a 200–300 token
window would have merged two or three unrelated student opinions into one chunk
and diluted retrieval. I changed the strategy to review-aware chunking (one
review per chunk, with the 280-token figure kept only as a safety cap) and
dropped cross-review overlap, then updated `planning.md` to record the change and
the reasoning. The divergence improved retrieval precision because each embedding
now represents exactly one opinion.

---

## AI Usage

**Instance 1 — Ingestion and chunking**

- *What I gave the AI:* My `planning.md` Documents section (10 cleaned `.txt`
  files, each with a header plus several "Review N" blocks) and my Chunking
  Strategy section, and asked it to implement `load_documents()`,
  `clean_text()`, and a review-aware `chunk_text()` that emits one chunk per
  review with source/professor/course metadata and prints sample chunks + a count.
- *What it produced:* A working `ingest.py` that parsed reviews on the "Review N"
  markers and attached metadata.
- *What I changed or overrode:* It initially leaned toward a generic fixed-size
  character splitter (matching my old 200–300 token plan). I overrode it to split
  on review boundaries instead, kept the 280-token figure only as a fallback cap,
  added a `len(chunk) > 0` empty-chunk filter, and made each chunk prefix the
  professor name so name-based queries are retrievable. I verified by printing 5
  chunks and confirming each was self-contained.

**Instance 2 — Grounded generation**

- *What I gave the AI:* My grounding requirement (answer only from retrieved
  context, decline otherwise), the desired output format (answer + programmatic
  source list), and the Gradio skeleton from the instructions, and asked it to
  implement `ask()` and `app.py`.
- *What it produced:* A `query.py` that built a context prompt and called Groq,
  and a Gradio app.
- *What I changed or overrode:* The first version let the LLM cite its own
  sources and relied solely on the prompt for grounding. I rewrote it so source
  attribution is built **programmatically** from retrieval metadata (the model
  never invents citations), strengthened the system prompt from "use the
  documents" to "use ONLY the CONTEXT ... do not use prior knowledge," and added
  an out-of-coverage distance guard (best distance > 0.85 → decline before
  calling the LLM). I confirmed an out-of-domain question ("best pizza near
  campus") correctly returns "I don't have enough information on that."
