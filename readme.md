# Diabetes RAG — ADA 2025 Clinical Q&A

A production-deployed **retrieval-augmented generation (RAG)** system for querying the [ADA Standards of Care in Diabetes 2025](https://diabetesjournals.org/care/issue/48/Supplement_1). Answers are grounded in guidelines, source-cited, and the system **abstains rather than hallucinates** when evidence is insufficient.

**Live demo → [diabetes-rag.onrender.com](https://diabetes-rag.onrender.com)**

![Diabetes RAG UI](https://raw.githubusercontent.com/Dhyey2901/diabetes-rag/main/docs/screenshot.png)

---

## What it does

- Accepts clinical questions about diabetes management
- Retrieves the most relevant chunks from 19 ADA 2025 guideline chapters using hybrid search
- Streams the answer token-by-token with bracketed citations
- Shows a confidence badge (green / amber / red) on every response
- Refuses to answer when retrieval confidence is below threshold (abstention-first design)
- Tracks usage and answer quality in a built-in analytics dashboard

---

## Architecture

```text
PDF (358 pages)
    │
    ▼
ingest.py  ──  TOC-based page extraction (PyMuPDF)
    │
    ▼
data/clean/  ──  19 Markdown files, one per ADA chapter
    │
    ▼
chunk.py  ──  sliding-window chunker  →  index/chunks.jsonl  (912 chunks)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
bm25.py                            embed_index.py
BM25Okapi sparse index             fastembed MiniLM-L6-v2
(term overlap)                     (semantic embeddings, ONNX)
    │                                      │
    └────────────────┬─────────────────────┘
                     ▼
               retrieve.py
       Weighted BM25 + dense fusion
       Reciprocal Rank Fusion (RRF)
       MMR diversification
       Section pinning (A1C / diet / kidney intents)
                     │
                     ▼
                   qa.py
       Context reducer (top-k re-scoring)
       OpenRouter LLM (cloud) / Ollama (local fallback)
       Lexical support check  →  abstain if overlap < 0.08
                     │
                     ▼
               web_ui.py
       Flask · SSE streaming · Confidence badge
       Citation pills · Analytics dashboard
```

---

## Tech Stack

| Layer | Tool |
| --- | --- |
| PDF extraction | PyMuPDF (`fitz`) — TOC page-based section splitting |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Dense retrieval | `fastembed` · `all-MiniLM-L6-v2` (ONNX, ~80 MB RAM) |
| Vector math | `numpy` — L2-normalised matmul cosine search |
| Hybrid fusion | Weighted score + Reciprocal Rank Fusion |
| Generation | OpenRouter API (cloud) · Ollama (local fallback) |
| Web framework | Flask 3 + gunicorn |
| Frontend | Bootstrap 5 · Chart.js · `marked.js` (markdown rendering) |
| Deployment | Render (free tier, auto-deploy from GitHub) |

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| fastembed over sentence-transformers | ONNX runtime uses ~80 MB vs ~450 MB (PyTorch) — fits Render's 512 MB free tier |
| TOC page-based ingestion | Text-matching section detection mis-assigns content when titles appear in cross-references; PDF bookmarks give exact page boundaries |
| Abstention-first | Confidence < 0.35 or lexical support < 0.08 → refuse rather than hallucinate |
| Two-pass retrieval | BM25 for exact term recall + dense for semantic; RRF normalises rank lists before fusion |
| MMR diversification | Prevents returning near-duplicate chunks from the same page |
| Section pinning | Regex detects A1C / diet / kidney intents → boosts relevant chapter scores before fusion |
| SSE streaming | Token-by-token delivery over `/stream` endpoint; falls back to `/ask` if SSE fails |

---

## Evaluation

84-question gold set with answerable/unanswerable split (`data/eval/gold_diabetes_80.json`):

| Metric | Score |
| --- | --- |
| Answerable accuracy | **86.7%** |
| Correct abstention rate | 58.3% |
| Overall accuracy | 76.2% |
| Avg retrieval confidence | 0.66 |
| Avg answer relevance | 0.50 |

**Strong categories:** cardiovascular risk, care coordination, education, lifestyle, monitoring, technology, weight management.

**Known limitation — abstention on out-of-scope queries:** Questions containing diabetes-adjacent terminology (e.g., drug dosing questions) trigger high retrieval confidence even when the specific answer isn't in the guidelines. Fixing this requires a query intent classifier or stricter cross-encoder reranker upstream of generation.

Run it yourself:

```bash
python src/evaluate_gold.py        # full 84 questions
python src/evaluate_gold.py 20     # quick 20-question sample
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [OpenRouter API key](https://openrouter.ai) (free) **or** [Ollama](https://ollama.com) running locally

```bash
# 1. Clone and install
git clone https://github.com/Dhyey2901/diabetes-rag.git
cd diabetes-rag
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY

# 3. Run (indexes already committed)
python -m flask --app "src.web_ui:create_app()" run
# or
gunicorn "src.web_ui:create_app()" --bind 0.0.0.0:5000 --workers 1
```

### Rebuild indexes from scratch

Only needed if you change the source PDF or chunking parameters:

```bash
python src/ingest.py        # PDF → data/clean/*.md  (19 chapters, TOC-based)
python src/chunk.py         # sliding-window chunks  →  index/chunks.jsonl
python src/bm25.py          # BM25 sparse index
python src/embed_index.py   # fastembed dense index  →  embeddings.npy
```

---

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | Required for cloud LLM. Get one free at openrouter.ai |
| `OPENROUTER_MODEL` | `google/gemma-3-27b-it:free` | OpenRouter model slug |
| `OLLAMA_MODEL` | `gemma:2b` | Local fallback model (when no API key is set) |
| `GEN_TOPK` | `4` | Chunks passed to the LLM as context |
| `CONF_ABSTAIN` | `0.35` | Retrieval confidence below which the system abstains |
| `SUPPORT_THRESHOLD` | `0.08` | Lexical overlap below which the LLM answer is discarded |
| `FLASK_SECRET_KEY` | random | Flask session signing key |

---

## Project Structure

```text
diabetes-rag/
├── data/
│   ├── raw/                      # Source PDF (not committed)
│   ├── clean/                    # 19 Markdown chapter files
│   └── eval/gold_diabetes_80.json
├── index/
│   ├── chunks.jsonl              # 912 chunks with section metadata
│   ├── bm25.pkl                  # Serialised BM25 index
│   ├── embeddings.npy            # L2-normalised MiniLM embeddings (912 × 384)
│   └── meta.jsonl
├── src/
│   ├── ingest.py                 # PDF → Markdown (TOC page-based)
│   ├── chunk.py                  # Sliding-window chunker
│   ├── bm25.py                   # BM25 index builder
│   ├── embed_index.py            # fastembed dense index builder
│   ├── retrieve.py               # Hybrid retriever (BM25 + dense + RRF + MMR)
│   ├── qa.py                     # QA pipeline (retrieval → LLM → QAResult)
│   ├── evaluate_gold.py          # Gold-set evaluator
│   └── web_ui.py                 # Flask app + streaming + metrics dashboard
├── .env.example
├── render.yaml                   # Render deployment config
├── Procfile
└── requirements.txt
```

---

## Limitations & Future Work

- **Table extraction**: ADA numeric targets appear in tables; PyMuPDF extracts these as plain text, losing row/column structure. A table-aware extractor (e.g. `camelot`) would improve coverage.
- **Cross-encoder reranker**: A `ms-marco-MiniLM` reranker between retrieval and generation would improve precision, especially for out-of-scope abstention.
- **Multi-turn context**: Session history is stored but not fed into retrieval — conversational follow-ups lose prior context.

---

## Data Use

This project uses the ADA Standards of Care 2025 for educational and research purposes only. For redistribution or derivative use of guideline text, follow [ADA's terms](https://diabetesjournals.org).
