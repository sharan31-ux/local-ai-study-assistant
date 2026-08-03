# Portfolio and interview notes

Notes for talking about this project. Everything here is factual — there are no invented benchmarks
or performance numbers, because none were measured.

---

## 30-second explanation

> It's a study assistant that turns my lecture PDFs into summaries, flashcards and practice quizzes,
> and lets me ask questions about them. What makes it different is that it runs completely locally:
> the language model is served by Ollama on my own machine, so no course material is ever uploaded
> anywhere. I built it in Python with Streamlit for the interface, and I implemented a lightweight
> RAG pipeline myself — PDF extraction, chunking, TF-IDF retrieval with cosine similarity — instead
> of using a framework, so I actually understand every stage.

---

## 2-minute explanation

> The problem I wanted to solve is that the tools students use to revise from lecture notes all
> require uploading those notes to someone else's server. For coursework that's often not allowed,
> and it's not free at scale. So I built a version that runs entirely on my laptop.
>
> The user uploads one or more PDFs, or pastes text. I extract the text with pypdf in memory —
> nothing is written to disk — then clean it, which mostly means normalising whitespace, stripping
> control characters and rejoining words that PDF hyphenation split across lines, while keeping
> paragraph breaks intact.
>
> There are four modes. Three of them — summary, flashcards and quiz — send the material to a local
> model with a prompt that specifies the output format. For flashcards and quizzes I ask for JSON,
> because I need structured data to render and export it, and I parse that defensively: local models
> often wrap JSON in Markdown fences or add a sentence before it, so I strip fences, slice out the
> JSON, and validate the shape. If a quiz question doesn't have exactly four choices, or the correct
> answer doesn't match one of them, I reject it rather than showing something broken. When parsing
> fails entirely I show the raw response instead of crashing.
>
> The fourth mode is question answering, and that's where retrieval comes in. Sending a whole
> textbook chapter to a 3B model doesn't fit in its context window, so I split the documents into
> overlapping chunks of about 1,500 characters, build a TF-IDF matrix over them with scikit-learn,
> and rank chunks against the question with cosine similarity. Only the top few chunks go to the
> model, with instructions to answer only from them and to say the answer wasn't found otherwise.
> I show the retrieved passages and their similarity scores in the UI, so you can see exactly what
> the answer was based on.
>
> I want to be precise about one thing: this is TF-IDF, which is lexical matching — it compares
> words, not meaning. A proper embedding model would handle paraphrasing better. I chose TF-IDF
> because it needs no model download, no vector database and no cost, and adding local embeddings
> through Ollama is the first improvement on my list.
>
> Architecturally, the Streamlit file only does presentation. All the logic lives in a package with
> separate modules for the Ollama client, document loading, text processing, retrieval, prompts and
> parsing, coordinated by a service layer. That's what lets me test the whole pipeline without
> Streamlit and without Ollama running — I inject a fake HTTP session into the client, so the test
> suite is fast, offline, and runs in GitHub Actions on every push.

---

## Key technical decisions

| Decision | Reasoning | What I gave up |
| --- | --- | --- |
| **Ollama over a hosted API** | Privacy is the point of the project; also free and works offline. | Output quality — a local 3B model is weaker than a large hosted one. |
| **No LangChain or LlamaIndex** | The RAG pipeline here is four steps. Writing them myself means I can explain each one, and the dependency list stays tiny. | Some convenience helpers I'd have got for free. |
| **TF-IDF instead of embeddings** | No model download, no vector store, no cost; scores are inspectable and easy to debug. | Semantic matching — paraphrased questions retrieve worse. |
| **Injected `requests.Session`** | Makes the Ollama client testable without patching modules or running a fake server. | One extra constructor parameter. |
| **Custom exception hierarchy** | Every error surfaces as a readable sentence in the UI with no stack trace, and the type tells the caller how to react. | More code than letting exceptions propagate. |
| **JSON output for flashcards and quizzes** | I need structured data to render, validate and export it. | Local models sometimes produce malformed JSON, so I had to write forgiving parsing. |
| **Truncating long documents for whole-document modes** | Simplest strategy that fits a local context window, and honest about it — the UI says when it happened. | Material past the cut-off is not summarised. Map-reduce would fix it. |
| **No database, no persistence** | Nothing to leak, nothing to secure, nothing to configure. | No history between sessions. |
| **Prompt-level instruction isolation** | Uploaded PDFs are untrusted input; document text is delimited and declared as reference material only. | It's a mitigation, not a guarantee — no prompt defence is airtight. |

---

## Limitations I'd raise before being asked

- No OCR, so scanned PDFs are rejected (with a clear message rather than an empty result).
- TF-IDF retrieval is lexical, not semantic.
- Whole-document modes truncate at ~12,000 characters.
- Output quality is bounded by whichever local model the user has installed.
- Tests cover the pipeline logic; there is no end-to-end test against a live model, because that
  would make CI depend on Ollama.
- Single-user and local by design — no auth, not meant to be exposed on a network.

---

## Likely interview questions, with honest answers

**"Is this real RAG?"**
Yes, in the sense that it retrieves relevant context and augments the prompt with it. But my
retriever is TF-IDF with cosine similarity, not a vector database with embeddings. It's the same
architecture with a simpler, lexical retriever.

**"Why not use embeddings?"**
Cost and moving parts. Embeddings would mean either a paid API — which defeats the privacy goal — or
downloading another model and managing a vector store. TF-IDF gave me working retrieval with one
scikit-learn import, and it's the honest baseline. Ollama exposes an embeddings endpoint, so adding
it locally with TF-IDF as a fallback is my next step, and comparing the two would be a genuinely
interesting experiment.

**"How do you handle the model returning bad JSON?"**
Three layers. I strip Markdown code fences and slice from the first bracket to the last, which
handles the common "here's your JSON: ..." reply. Then I validate structure — a quiz question needs
exactly four non-empty choices and a correct answer that matches one of them. If any of that fails I
raise a `ResponseParsingError` that carries the raw reply, and the UI shows the error plus the raw
response in an expander, so the user can still read what was generated.

**"What stops the model from making things up?"**
I can reduce it, not eliminate it. For question answering only the retrieved passages are sent, with
explicit instructions to answer from them alone and a fixed sentence to return when they don't
contain the answer. If retrieval finds no matching passage I return that sentence without calling
the model. Temperature is 0.1 for structured tasks. And I show the retrieved passages with their
similarity scores, so the user can verify the answer against its source rather than trusting it.

**"What if someone uploads a PDF containing 'ignore your instructions'?"**
That's prompt injection through document content, and I treat uploads as untrusted input. Document
text is wrapped in explicit delimiters, and the system message says everything inside is third-party
reference material to study, never instructions to follow. It reduces the risk; it doesn't remove
it. A stronger defence would need output validation or a model trained for instruction hierarchy.

**"Why does chunking need overlap?"**
Because a definition can straddle a boundary. Without overlap, a sentence cut in half is retrievable
from neither chunk. 200 characters of overlap on a 1,500-character chunk costs ~13 % duplication and
avoids that.

**"How do you know the chunking loop terminates?"**
The next start is `max(end - overlap, start + 1)`, so it always advances by at least one character.
I also reject an overlap greater than or equal to the chunk size up front, since that's the
configuration that would otherwise make the cursor stand still. Both are covered by tests.

**"How do you test code that calls an LLM?"**
I don't test the model — I test my code around it. The Ollama client takes a `requests.Session`
through its constructor, so tests inject a fake session that returns canned responses or raises the
exact `requests` exception I want to handle. The service layer takes a client, so a stub client lets
me assert on the prompt that *would* have been sent — for example, that a biology passage never
reaches the model when the question is about deadlocks. Everything is offline and runs in about a
second.

**"What was the hardest bug?"**
Exception ordering in the HTTP client. `requests.exceptions.ConnectTimeout` inherits from both
`ConnectionError` and `Timeout`, so with the `except` blocks in the wrong order every connect
timeout was reported as "Ollama is not running" — a misleading message that would send a user
looking in the wrong place. There's now a test pinning the ordering.

**"What would you do differently with more time?"**
Map-reduce summarisation instead of truncation, local embeddings alongside TF-IDF so I could measure
the difference on my own material, and streaming responses so long generations don't look frozen.

---

## Skills demonstrated

- **Python** — type hints throughout, dataclasses, custom exception hierarchy, `pathlib`, logging.
- **Software architecture** — layered design with presentation, orchestration, domain and
  infrastructure separated; dependency injection for testability.
- **Information retrieval** — chunking with overlap and boundary detection, TF-IDF vectorisation,
  cosine similarity ranking, top-*k* selection.
- **LLM integration** — prompt design for grounding and structured output, temperature control,
  defensive parsing of unreliable model responses, prompt-injection mitigation.
- **Testing** — 82 unit tests with dependency injection and fakes; no network, no model, no external
  fixtures.
- **Error handling and UX** — every failure path produces an actionable message instead of a stack
  trace.
- **Tooling and CI** — ruff, pytest, GitHub Actions, Makefile, editable package install with a
  `src/` layout.
- **Privacy engineering** — in-memory processing, no persistence, no telemetry, careful logging.

---

## CV project entry

> **Local AI Study Assistant** — Python, Streamlit, Ollama, scikit-learn
> Built a privacy-focused study tool that converts lecture PDFs into summaries, flashcards and
> multiple-choice quizzes, and answers questions about them, using a language model running locally
> via Ollama so no document leaves the user's machine. Implemented the retrieval-augmented
> generation pipeline from scratch — in-memory PDF extraction, text cleaning, overlapping chunking,
> and TF-IDF retrieval with cosine similarity — instead of using an LLM framework. Designed a
> layered architecture that isolates the UI from domain logic, with dependency injection enabling
> 82 unit tests that run offline without the model server, plus a GitHub Actions pipeline running
> ruff and pytest on every push.

---

## LinkedIn project description

> Local AI Study Assistant — a privacy-focused study tool that turns lecture PDFs into summaries,
> flashcards, practice quizzes and grounded answers, powered by a language model running entirely on
> your own machine through Ollama. No API keys, no cloud, no data leaving your laptop.
>
> I implemented the RAG pipeline from scratch rather than using a framework: PDF extraction with
> pypdf, text cleaning, overlapping chunking, and TF-IDF retrieval with cosine similarity in
> scikit-learn — lexical retrieval, not embeddings, which keeps it free and fully explainable.
> The architecture separates the Streamlit UI from the domain logic, so the whole pipeline is unit
> tested offline with no model server required, and CI runs lint and tests on every push.
>
> Python · Streamlit · Ollama · scikit-learn · pytest · GitHub Actions
