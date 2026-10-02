# Tesla Intelligence — RAG over Tesla's SEC Filings

A local-first Retrieval-Augmented Generation (RAG) system that answers questions about Tesla using only Tesla's own Form 10-K filings (FY2023–FY2025), with every answer grounded in cited source chunks.

Every answer comes with its source: filing year, page number, and chunk ID — the model is instructed to say "the available documents do not contain enough information" rather than guessing when the filings don't cover something.

## Architecture

**Offline ingestion pipeline** (run once, builds the index):

```
Tesla 10-K PDFs (2023 / 2024 / 2025)
        ↓  PyMuPDF — page-by-page extraction, page markers preserved
Raw extracted text
        ↓  regex cleanup — SEC footer/URL removal, whitespace normalization
Cleaned text
        ↓  paragraph + sentence aware chunking, 1,500-char target, no overlap
1,134 chunks
        ↓  sentence-transformers/all-MiniLM-L6-v2 — 384-dim embeddings
Embedded vectors
        ↓  FAISS IndexFlatIP (cosine similarity via L2-normalization)
Searchable vector index
```

**Online query pipeline** (runs per question):

```
User question (browser)
        ↓  POST /ask
FastAPI backend
        ↓  embed question → FAISS top-5 search → build context
Prompt (instruction + 5 retrieved chunks)
        ↓
Groq API — openai/gpt-oss-120b
        ↓
Grounded answer + cited sources → rendered in the browser
```

## Tech Stack

| Layer | Choice |
|---|---|
| PDF extraction | PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), 384-dim, runs locally |
| Vector store | FAISS (`IndexFlatIP`) |
| LLM | Groq API, `openai/gpt-oss-120b` |
| Backend | FastAPI |
| Frontend | Vanilla HTML / CSS / JS |

## Project Structure

```
Tesla-RAG/
├── main.py                  # FastAPI backend — /ask endpoint, retrieval + generation
├── extract_text.py          # Step 1: PDF → raw text (PyMuPDF)
├── clean_text.py             # Step 2: text cleanup
├── chunk_text.py             # Step 3: structure-aware chunking
├── create_embeddings.py      # Step 4: local embedding generation
├── build_faiss_index.py      # Step 5: FAISS index build
├── search.py                 # Standalone retrieval test script
├── rag.py                    # Standalone end-to-end RAG test script
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── data/
    ├── raw/                  # Source Tesla 10-K PDFs
    ├── processed/            # Extracted text
    ├── cleaned/               # Cleaned text
    ├── chunks_v3/             # Final chunks (1,134 total)
    ├── embeddings/            # Chunk embeddings
    └── faiss/                 # tesla.index + metadata.json
```

## Running It Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install pymupdf sentence-transformers faiss-cpu groq fastapi uvicorn

export GROQ_API_KEY="your-groq-api-key"

uvicorn main:app --reload
```

Then open `frontend/index.html` in a browser.

## Why 10-K Filings

Form 10-Ks are the annual report every U.S. public company is legally required to file with the SEC — audited, standardized in structure (same numbered Items every year), and freely available. Using three consecutive fiscal years lets the system handle not just single-fact lookups but year-over-year comparison questions that require combining multiple retrieved chunks.

## Example

**Q: What was Tesla's total revenue in 2024?**
**A:** Tesla's total revenue for the year ended December 31, 2024 was $97,690 million.
*Source: Tesla 2024 10-K, Page 53, Chunk 177*

## Notes

- No API keys are stored in this repository — `GROQ_API_KEY` is read from the environment at runtime.
- This is a local demo project with no deployment target; it is designed to run entirely on a single machine.
