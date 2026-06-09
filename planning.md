# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

This system covers **student reviews of Computer Science professors at Hunter College (CUNY)**, sourced from RateMyProfessors. Without this kind of crowd-sourced feedback, it is hard for students to gauge a professor's exam difficulty, grading fairness, workload, lecture clarity, and accessibility before registering. The official CUNY course catalog only lists titles and prerequisites — it says nothing about *how* a professor teaches, whether exams are curved, or whether office hours are useful. That experiential knowledge lives in scattered, JavaScript-rendered review pages that are hard to search, compare, or summarize across professors. A retrieval system that grounds answers in the actual review text lets a student ask a natural-language question ("How hard are Professor Shostak's exams?") and get a sourced, balanced answer.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

All sources are RateMyProfessors review pages for CS professors at Hunter College. Each page was fetched, cleaned of site boilerplate (nav, footers, "Helpful" counts, ads), and saved as a plain `.txt` file in `documents/`. The professors span a wide range of ratings (1.2 to 4.3 / 5) and subtopics — intro courses, operating systems, ML, security — so the corpus covers different perspectives.

| # | Professor | Local file | URL |
|---|-----------|------------|-----|
| 1 | Pavel Shostak | `documents/rmp_shostak.txt` | https://www.ratemyprofessors.com/professor/1823870 |
| 2 | Saad Mneimneh | `documents/rmp_mneimneh.txt` | https://www.ratemyprofessors.com/professor/926045 |
| 3 | Tong Yi | `documents/rmp_yi.txt` | https://www.ratemyprofessors.com/professor/2634841 |
| 4 | Melissa Lynch | `documents/rmp_lynch.txt` | https://www.ratemyprofessors.com/professor/2505090 |
| 5 | Saptarshi Debroy | `documents/rmp_debroy.txt` | https://www.ratemyprofessors.com/professor/2220139 |
| 6 | Sven Dietrich | `documents/rmp_dietrich.txt` | https://www.ratemyprofessors.com/professor/2674099 |
| 7 | Susan Epstein | `documents/rmp_epstein.txt` | https://www.ratemyprofessors.com/professor/192300 |
| 8 | Raffi Khatchadourian | `documents/rmp_khatchadourian.txt` | https://www.ratemyprofessors.com/professor/2259095 |
| 9 | Raj Korpan | `documents/rmp_korpan.txt` | https://www.ratemyprofessors.com/professor/2659561 |
| 10 | Mohammad Manshaei | `documents/rmp_manshaei.txt` | https://www.ratemyprofessors.com/professor/3066287 |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Strategy: chunk per review (review-aware splitting), with a token cap.**

**Chunk size:** one student review per chunk, capped at ~280 tokens. Each chunk
is prefixed with a context line (professor name + course + quality/difficulty
ratings) so it is self-contained. If a single review exceeds the 280-token cap,
it is further split with a word-window splitter (~250 tokens, 30-token overlap).

**Overlap:** 0 tokens between separate reviews (reviews are independent
opinions, so overlap would only duplicate unrelated content); 30-token overlap
only *within* an oversized review that has to be sub-split.

**Why this fits the documents (revised after inspecting the corpus):** My
original plan was a fixed 200–300 token window with 30-token overlap. After
loading the documents I saw each RateMyProfessors review is a short, complete
opinion (~40–90 tokens) and each file holds ~3–5 of them. A fixed 200–300 token
window would merge two or three *unrelated* student opinions into one chunk,
diluting the embedding and hurting precise retrieval. Splitting on the natural
review boundary keeps each chunk a single "complete thought" (e.g. "Shostak's
exams are 90% of the grade and require memorizing exact definitions"), which is
exactly what semantic search needs to match a specific question. Prefixing the
professor name makes name-based queries retrievable even when the review text
itself never repeats the name.

**Expected chunk count:** ~48 chunks across the 10 documents (one per review).

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. It runs
locally with no API key or rate limit, produces 384-dim embeddings, and is fast
enough to embed the whole corpus in seconds. (Note: Groq's
`llama-3.3-70b-versatile` is the *generation* LLM in Milestone 5 — it does **not**
do the embedding. These are two separate models.)

**Vector store:** ChromaDB (persistent local collection), storing each chunk's
text, embedding, and metadata (source filename, professor name, course, chunk
index). Chroma returns cosine distance, which I use as a relevance signal.

**Top-k:** 5 chunks per query. Five is enough to surface a few independent
reviews about the same professor so the answer reflects more than one opinion,
without pulling in loosely related reviews from other professors that would pull
the LLM off-target. Too few (k=1–2) risks missing the relevant review entirely
if the best match isn't rank 1; too many (k=10+) floods the prompt with
off-professor noise.

**Why semantic search works here:** queries rarely use the exact words of a
review ("How hard are the exams?" vs. a review saying "his tests are strict and
90% of the grade"). Embeddings map both into nearby vectors based on meaning, so
the relevant chunk is retrieved even without shared keywords.

**Production tradeoff reflection:** If cost weren't a constraint and this served
real students, I'd weigh: (1) **accuracy on domain-specific text** — a larger
model like OpenAI `text-embedding-3-large` or a fine-tuned model would better
distinguish near-synonyms ("curved" vs. "lenient"); (2) **context length** —
MiniLM truncates at 256 tokens, fine for short reviews but limiting if I later
ingest long syllabi; (3) **multilingual support** — several reviews mention
non-native-English students, so a multilingual model could help if reviews were
written in other languages; (4) **latency vs. local control** — MiniLM is local
and instant, while a hosted API adds network latency and a dependency, but scales
better and offloads memory. For this corpus the accuracy gain wouldn't justify
the cost/latency, so MiniLM is the right default.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer (verifiable against the reviews) |
|---|----------|--------------------------------------------------|
| 1 | What do students say about the exams in Professor Pavel Shostak's classes? | Exams are about 90% of the grade, are strict, and require memorizing his specific definitions/terms — he looks for exact words in answers. Reviews split between "easy if you memorize definitions" and "you learn nothing." |
| 2 | How heavy is the workload and difficulty in Professor Saad Mneimneh's CSCI150 class? | A genuinely hard class with a heavy workload — students report studying 1–2 weeks, 3–8 hours/day for exams, and difficult (sometimes ungraded) homework. But he curves exams, provides notes/slides, and is accessible, so the difficulty is attributed to the material, not the professor. |
| 3 | Why do students find Professor Tong Yi's lectures hard to follow? | Her accent / English makes the lectures hard to understand, so students call the lectures "useless" and rely on tutoring, recitations, and the past exams she posts online. She's also described as a tough grader. |
| 4 | What do reviews say about Professor Melissa Lynch's punctuality and responsiveness to emails? | She frequently shows up to class ~30 minutes late, is slow to return grades (can take ~2 weeks), and does not respond to emails. A minority still like her partial-credit grading and teaching style. |
| 5 | What is unusual about Professor Sven Dietrich's exams and quizzes? | Unusual exam logistics: pen only, assigned seating, and individually printed exams per student; pop quizzes test random material (including a movie students had to watch); multiple-choice quizzes can be multi-answer with no partial credit. He does give a good curve, but lectures are mostly reading slides. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Biased or one-sided reviews**: A single disgruntled (or overly happy) student
   can skew results. Mitigation: retrieve multiple chunks to balance opposing
   views and weight recent reviews more heavily.

2. **Spam and irrelevant reviews**: Troll posts or reviews from students who
   didn't complete the course. Mitigation: filter reviews by upvote count
   and set a relevance threshold (>0.65 cosine similarity) during retrieval.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```
┌──────────────────────┐     ┌──────────────────────┐     ┌────────────────────────────┐
│  1. Document          │     │  2. Chunking          │     │  3. Embedding + Vector Store │
│     Ingestion         │     │                       │     │                              │
│  documents/*.txt   ───┼────▶│  ingest.py            │────▶│  all-MiniLM-L6-v2            │
│  (cleaned RMP reviews)│     │  one chunk per review │     │  (sentence-transformers)     │
│  loaded from disk     │     │  + prof/course prefix │     │         │                    │
└──────────────────────┘     │  ~280-token cap       │     │         ▼                    │
                             └──────────────────────┘     │  ChromaDB (persistent)       │
                                                          │  + metadata: source, prof,   │
                                                          │    course, chunk index       │
                                                          └──────────────┬───────────────┘
                                                                         │
                          ┌──────────────────────────────┐              │
                          │  5. Generation                │              ▼
   user question  ───────▶│  Groq llama-3.3-70b-versatile │◀──┌────────────────────────────┐
                          │  grounded prompt (context     │   │  4. Retrieval                │
   answer + sources ◀─────│  only) + source attribution   │◀──│  query → embed → top-k=5     │
                          │  Gradio web UI (app.py)       │   │  cosine distance from Chroma │
                          └──────────────────────────────┘   │  (retrieve.py)               │
                                                              └────────────────────────────┘
```

**Stage → tool/library:**
1. Ingestion — Python file I/O reading `documents/*.txt`
2. Chunking — custom review-aware splitter in `ingest.py`
3. Embedding + Vector Store — `all-MiniLM-L6-v2` (sentence-transformers) → ChromaDB
4. Retrieval — ChromaDB cosine similarity, top-k = 5 (`retrieve.py`)
5. Generation — Groq `llama-3.3-70b-versatile`, grounded prompt, Gradio UI (`query.py`, `app.py`)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->
     

**Milestone 3 — Ingestion and chunking:**
Tool: Cursor's AI assistant (Claude). Input: my Documents section (10 cleaned
`.txt` files, one per professor, with a header + several "Review N" blocks) and
my Chunking Strategy section (chunk per review, ~280-token cap, prof/course
prefix, no cross-review overlap). Expected output: a `ingest.py` with
`load_documents()`, `clean_text()`, and `chunk_text()` that produces one chunk
per review, attaches metadata (source filename, professor, course, chunk index),
filters empty chunks, and prints 5 sample chunks + a total count. Verification:
I'll print 5 chunks and confirm each is self-contained, readable, and tagged with
the right source; I'll confirm the count is in the 40–60 range, not <50 (too
large) or >2000 (too small).

**Milestone 4 — Embedding and retrieval:**
Tool: Claude. Input: my Retrieval Approach section (all-MiniLM-L6-v2, ChromaDB,
top-k=5, metadata fields) plus the architecture diagram. Expected output:
`embed.py` that loads chunks from the ingestion step, embeds them with
`SentenceTransformer("all-MiniLM-L6-v2")`, and upserts them into a persistent
ChromaDB collection with metadata; and `retrieve.py` with a `retrieve(query, k=5)`
function returning chunks + source + distance. Verification: I'll run 3 of my 5
eval queries, print returned chunks and distance scores, and confirm top results
are on-topic with distance < 0.5; if any ChromaDB API call is unfamiliar I'll ask
the AI to explain it.

**Milestone 5 — Generation and interface:**
Tool: Claude. Input: my grounding requirement (answer from retrieved context
only, say "I don't have enough information" otherwise), desired output format
(answer + bulleted source list), and the Gradio skeleton from the instructions.
Expected output: `query.py` with an `ask(question)` function that retrieves
top-k, builds a grounded prompt, calls Groq `llama-3.3-70b-versatile`, and
returns `{"answer": ..., "sources": [...]}` where sources are appended
programmatically (not left to the LLM); and `app.py` with the Gradio Blocks UI.
Verification: before running I'll read the system prompt to confirm it *enforces*
("use only the provided context") rather than suggests grounding, and confirm
sources come from retrieval metadata. I'll test an out-of-coverage question and
confirm the system declines instead of hallucinating.
