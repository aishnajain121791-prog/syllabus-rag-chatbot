# Syllabus Q&A Chatbot (RAG)

A standalone Python Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from uploaded course syllabus and lecture-note PDFs, with page-level source citations, SQLite-backed chat & note persistence, study tools, and a Streamlit web interface.

---

## Architecture

```text
PDFs (data/)
  │
  ▼  pypdf (inspect_pdf + page-by-page extraction + ~500-char chunks, 100-char overlap)
Chunks + Metadata  ──►  {"text", "source", "page"}
  │
  ▼  sentence-transformers ("all-MiniLM-L6-v2", L2-normalized, cached via st.cache_resource)
Embeddings
  │
  ▼  FAISS IndexFlatIP (cosine similarity) + persisted metadata (faiss_index/)
Top-k Retriever (global or filtered by IDSelectorBatch)
  │
  ├──►  MIN_SCORE (0.30) gate: if top score < 0.30, skip LLM & return "I couldn't find this in your notes"
  │
  ├──►  SQLite (app.db): notes, chat_sessions, messages (with JSON sources)
  │
  ▼  Last 4 messages (follow-up rewrite) + Filtered Context + Grounding Prompt
Gemini LLM (google-genai)  ──►  Grounded Answer + Source Citations
```

1. **Document Ingestion (`src/loader.py`)**: `inspect_pdf()` validates uploaded PDFs (catching empty, scanned/image-only, and corrupted files), while `load_pdfs()` extracts text page by page (`1`-indexed) with `pypdf`, normalizes whitespace, and splits text into overlapping ~500-character chunks (100-character overlap) at word boundaries.
2. **Embedding & Vector Indexing (`src/retriever.py`)**: Encodes chunks with `all-MiniLM-L6-v2` (`normalize_embeddings=True`) and builds a `faiss.IndexFlatIP` index persisted to `faiss_index/index.faiss` and `faiss_index/chunks.json`. `clear_index()` resets both disk files and the in-memory cache when all notes are removed.
3. **Database Persistence (`src/database.py`)**: Built-in `sqlite3` database (`app.db`, ignored by Git) with foreign keys (`ON DELETE CASCADE`) and three tables:
   - `notes(id, filename, subject, uploaded_at, num_pages, num_chunks)`
   - `chat_sessions(id, title, created_at)`
   - `messages(id, session_id, role, content, sources, created_at)`
4. **Conversational RAG Pipeline (`src/rag.py`)**: Uses up to the last 4 chat messages to rewrite follow-up questions (e.g., *"explain that more simply"*) into standalone queries before retrieval, supports optional note/subject filtering via `faiss.IDSelectorBatch`, enforces a configurable `MIN_SCORE = 0.30` similarity threshold (logging top scores and skipping the LLM when retrieval falls below threshold), uses the `google-genai` SDK (`gemini-3.6-flash` by default, configurable via `GEMINI_MODEL`), and surfaces clear messages for missing keys, missing models, or API errors.
5. **Web Interface (`app.py`)**: Two-page Streamlit app (**Chat** and **My Notes**) with `@st.cache_resource` caching for the `SentenceTransformer` model:
   - **Chat**: `st.chat_message` / `st.chat_input` layout, sidebar session history (**➕ New chat**, reload previous chats titled from the first question, delete session), **Search scope** dropdown (`All notes` plus user-uploaded subjects/files), study buttons (**Summarize this note**, **Generate 5 quiz questions** with answers hidden in an expander, **Key terms**), and **Download chat as .txt**.
   - **My Notes**: Multi-PDF uploader with subject tagging, duplicate-upload handling (`upsert_note`), empty/scanned/corrupted PDF rejection, notes table (`Name`, `Subject`, `Date`, `Pages`), and per-note deletion with automatic index rebuild.

---

## Project Structure

```text
syllabus-rag/
├── app.py                      # Streamlit web application (Chat & My Notes)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Excludes .env, venv/, FAISS index files, and app.db
├── data/                       # Syllabus and lecture-note PDFs
├── src/
│   ├── loader.py               # PDF validation, loading, and overlapping chunking
│   ├── retriever.py            # Sentence-Transformers + FAISS IndexFlatIP search
│   ├── database.py             # SQLite persistence for notes, sessions, and messages
│   └── rag.py                  # Follow-up rewriting, MIN_SCORE gate, filtered retrieval, Gemini Q&A
├── eval/
│   ├── questions.json          # 25 answerable Q&A pairs + 5 unanswerable trap questions
│   ├── run_eval.py             # Hit@1, Hit@3, MRR, chunk accuracy, and trap refusal evaluation
│   ├── results.md              # Latest evaluation report
│   ├── chunk_experiment.py     # Grid search over chunk_size and overlap on temp indexes
│   └── chunk_results.md        # Chunk-size experiment table
└── tests/
    ├── test_loader.py          # Prints 5 sample chunks from data/
    ├── test_retriever.py       # Tests top-3 semantic search on a sample query
    └── test_database.py        # Tests all SQLite CRUD helpers in src/database.py
```

---

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Gemini API key

Copy `.env.example` to `.env` and replace `your_key_here` with your actual key:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 4. Add PDFs

Place syllabus or lecture-note PDFs inside `data/` or upload them in the **My Notes** page of the Streamlit UI.

---

## How to Run

### Launch the Streamlit Chatbot

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (default `http://localhost:8501`):
- Switch between **Chat** and **My Notes** from the sidebar.
- In **My Notes**, enter a subject name, upload one or more text-based PDFs, or delete existing notes.
- In **Chat**, pick **All notes** or a specific user-uploaded subject/file under **Search scope**, ask follow-up questions, run **Summarize this note** / **Generate 5 quiz questions** / **Key terms**, or export the conversation with **Download chat as .txt**.

### Run Standalone Tests & Evaluation

```bash
python tests/test_loader.py      # Verify PDF text extraction and chunking (prints 5 sample chunks)
python tests/test_retriever.py   # Build/load FAISS index and print top-3 hits for "What topics are in Unit 3?"
python tests/test_database.py    # Verify SQLite notes, chat_sessions, and messages CRUD
python eval/run_eval.py          # Run the 30-question evaluation (writes eval/results.md)
python eval/chunk_experiment.py  # Compare chunk_size/overlap pairs (writes eval/chunk_results.md)
```

---

## Example Questions

- *"What topics are in Unit 3?"*
- *"What are the objectives of the Computer Organization (R17A0510) course?"*
- *"What is cache memory and what property of computer programs makes it effective?"*
- *"Explain that more simply"* (follow-up rewritten using the last 4 messages)
- *"What is a DMA controller and how does the CPU initialize it?"*

---

## Chunk-Size Experiment (`eval/chunk_results.md`)

Evaluated across temporary FAISS indexes (without modifying `faiss_index/`) on the initial 10 answerable questions in `eval/questions.json` using top-3 `IndexFlatIP` retrieval:

| Chunk Size | Overlap | Total Chunks | Hit@1 | Hit@3 | MRR | Expected-Chunk Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 100 | 1249 | 90.0% (9/10) | 100.0% (10/10) | 0.9500 | 40.0% (12/30) |
| 500 | 150 | 1371 | 90.0% (9/10) | 100.0% (10/10) | 0.9500 | 43.3% (13/30) |
| 800 | 100 | 771 | 50.0% (5/10) | 80.0% (8/10) | 0.6500 | 30.0% (9/30) |
| 800 | 150 | 812 | 60.0% (6/10) | 80.0% (8/10) | 0.6833 | 26.7% (8/30) |
| 1200 | 100 | 547 | 50.0% (5/10) | 60.0% (6/10) | 0.5333 | 20.0% (6/30) |
| 1200 | 150 | 553 | 50.0% (5/10) | 60.0% (6/10) | 0.5333 | 20.0% (6/30) |

---

## Evaluation Results (`eval/results.md`)

Evaluated on the extended **30-question** benchmark in `eval/questions.json` (**25** answerable questions including 15 paraphrased/harder items + **5** out-of-domain trap questions) with `MIN_SCORE = 0.30`:

| Metric | Result |
| --- | --- |
| **Hit@1** | `72.0%` (`18/25`) |
| **Hit@3** | `84.0%` (`21/25`) |
| **MRR** | `0.7733` |
| **Expected-Chunk Accuracy** | `33.3%` (`25/75`) |
| **Trap Refusal Rate** | `20.0%` (`1/5`: `0` threshold, `1` LLM, `0` answered, `4` `429` API errors) |

**Notes on the evaluation run:**
- Adding the 15 paraphrased and harder questions lowered Hit@1 from `90.0%` to `72.0%` and Hit@3 from `100.0%` to `84.0%` (4 harder questions missed Hit@3 due to topic overlap across the three PDFs or non-verbatim phrasing).
- All 5 trap questions scored between `0.315` and `0.516` on `all-MiniLM-L6-v2`, so none were blocked by `MIN_SCORE = 0.30`. Trap 1 (Adam optimizer, `0.315`) was cleanly refused by Gemini (`"I don't know..."`), while Traps 2–5 (and 5 answerable calls) hit `429 RESOURCE_EXHAUSTED` rate limits from 30 back-to-back API calls.

---

## Design Decisions

- **Chunk size (`500` chars) & overlap (`100` chars)**: Kept `chunk_size=500, overlap=100` as defaults in `src/loader.py`. In the experiment above, `500`-character chunks dramatically outperformed `800` and `1200` (`90%` vs `50–60%` Hit@1 and `0.9500` vs `0.5333–0.6833` MRR) because `all-MiniLM-L6-v2` embeds short, single-topic windows cleanly whereas larger chunks blend multiple topics and dilute vector similarity. `overlap=100` preserves sentence context across boundaries while keeping `~9%` fewer total chunks than `overlap=150` (`1,249` vs `1,371`).
- **Similarity threshold (`MIN_SCORE = 0.30`)**: Added as a single tunable constant at the top of `src/rag.py` with score logging. Completely unrelated queries (e.g., baking recipes scoring `0.1229`) are rejected immediately without spending an LLM call, while valid syllabus queries like `"What topics are in Unit 3?"` (`0.3744`) still pass through. Because trap evaluation showed unrelated technical questions can still score `0.315–0.516`, `0.30` acts as a conservative low-end floor backed up by the LLM's strict *"I don't know"* instruction.

---

## Limitations

- **Tables and diagrams are not handled**: `pypdf` only extracts plain text streams; circuit/architecture diagrams are ignored, and multi-column tables (such as syllabus schedules or instruction tables) can lose row/column alignment.
- **Depends on retrieval quality**: Because Gemini is instructed to answer strictly from the retrieved `k=3` chunks, missed retrievals (seen on `4/25` harder questions in `eval/results.md`) or fixed 500-character splits across headings can cause incomplete answers or *"I don't know"* responses.
- **Scanned PDFs are unsupported**: Image-only or scanned PDFs contain no embedded text layer for `pypdf` to extract; `inspect_pdf()` detects and skips them with a UI warning rather than running OCR.
- **Embedding threshold sensitivity & API rate limits**: `all-MiniLM-L6-v2` cosine scores on out-of-domain questions sharing words like `"CPU"` or `"programs"` can exceed `0.30` (e.g., `0.516` on the Python GIL trap), and free-tier Gemini rate limits (`429`) or temporary model load spikes (`503`/`504`) can interrupt rapid batch runs.

---

## Future Improvements

- **Reranking**: Retrieve a wider candidate set (e.g., top `10`–`20` chunks from FAISS) and apply a cross-encoder reranker to select the strongest chunks before calling the LLM.
- **Hybrid BM25 search**: Combine sparse keyword search (BM25) with dense `all-MiniLM-L6-v2` embeddings (e.g., via Reciprocal Rank Fusion) to improve exact matching on course codes (`R17A0510`), unit numbers, and technical acronyms.
- **OCR support for scanned PDFs**: Add an optional OCR fallback (e.g., Tesseract or Vision LLM extraction) for scanned lecture notes.
