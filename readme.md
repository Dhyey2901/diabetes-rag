# Diabetes Clinical Q&A — ADA 2025 RAG System

An evidence-grounded retrieval-augmented generation (RAG) system for querying the [ADA Standards of Care in Diabetes 2025](https://diabetesjournals.org/care/issue/48/Supplement_1). Built end-to-end with a hybrid BM25 + dense retriever, local LLM generation via Ollama, and a Flask web UI with a live metrics dashboard.

> **Design principle:** the system abstains rather than hallucinates. If the evidence is not clearly present in the loaded corpus, it says so.

---

## Architecture

```text
PDF ──► ingest.py ──► data/clean/*.md   (19 ADA chapters, one file each)
                           │
                       chunk.py ──► index/chunks.jsonl   (500-word overlapping chunks)
                           │
             ┌─────────────┴──────────────┐
          bm25.py                   embed_index.py
        (BM25Okapi)          (MiniLM-L6-v2 · numpy cosine)
             │                           │
             └─────────┬─────────────────┘
                   retrieve.py
            Weighted fusion + RRF + MMR
                       │
                     qa.py
         Context reducer ──► Ollama LLM
                       │
               QAResult (answer + citations + confidence)
                       │
               web_ui.py  /  CLI
```

**Key design decisions:**

| Decision | Why |
| --- | --- |
| Numpy matmul instead of Annoy | Annoy segfaults on Python 3.14 / Apple Silicon; numpy L2-normalised dot product is identical and portable |
| Two-pass retrieval | BM25 for exact term coverage + dense for semantic; weighted fusion + RRF surfaces both |
| MMR diversification | Prevents returning near-duplicate chunks from the same page |
| Abstention threshold | Confidence below 0.35 or lexical support overlap below 0.08 → safe refusal |
| Section pinning | Regex detects A1C / diet / kidney intents → boosts matching chapter scores |

---

## Tech Stack

| Layer | Library / Tool |
| --- | --- |
| PDF extraction | PyMuPDF (`fitz`) |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Dense retrieval | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector math | `numpy` (cosine search via matmul) |
| Generation | Ollama HTTP API (default: `gemma:2b`) |
| Web UI | Flask + Bootstrap 5 + Chart.js |
| Security | `markupsafe.escape()` on all user/LLM output |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally (`brew install ollama && ollama pull gemma:2b`)

```bash
# 1. Clone and install
git clone https://github.com/Dhyey2901/diabetes-rag.git
cd diabetes-rag
pip install -r requirements.txt

# 2. Copy env template
cp .env.example .env   # edit if needed

# 3. Run (indexes already committed)
python src/run_generic_rag.py --test           # 5-question smoke test
python src/run_generic_rag.py "A1C target for adults with type 2 diabetes"
python src/run_generic_rag.py --web            # web UI at http://localhost:5000
```

### Rebuild indexes from scratch (optional)

Only needed if you swap the PDF or edit the Markdown.

```bash
python src/ingest.py        # PDF → data/clean/*.md  (19 chapters)
python src/chunk.py         # chunks.jsonl
python src/bm25.py          # BM25 index
python src/embed_index.py   # embeddings.npy
```

---

## Project Structure

```text
diabetes-rag/
├── data/
│   ├── raw/                     # ADA PDF (not committed; add your own)
│   ├── clean/                   # Extracted Markdown (one file per ADA chapter)
│   └── eval/
│       └── gold_diabetes_80.json
├── index/
│   ├── chunks.jsonl             # All chunks with metadata
│   ├── bm25.pkl                 # Serialised BM25 index
│   ├── bm25_meta.jsonl
│   ├── embeddings.npy           # L2-normalised MiniLM embeddings
│   └── meta.jsonl
├── results/
│   └── evaluation_results.json
├── src/
│   ├── ingest.py                # PDF extraction (PyMuPDF)
│   ├── chunk.py                 # Sliding-window chunker
│   ├── bm25.py                  # BM25 index builder
│   ├── embed_index.py           # Dense embedding builder
│   ├── retrieve.py              # Hybrid retriever (BM25 + dense + fusion + MMR)
│   ├── qa.py                    # Full QA pipeline (retrieval → LLM → QAResult)
│   ├── evaluate_gold.py         # 80-question gold-set evaluator
│   ├── web_ui.py                # Flask app + metrics dashboard
│   └── run_generic_rag.py       # CLI entrypoint
├── .env.example
├── requirements.txt
└── readme.md
```

---

## Evaluation

Run the 80-question gold set (answerable + unanswerable split):

```bash
python src/run_generic_rag.py --evaluate
```

Results are saved to `results/evaluation_results.json`. Key metrics reported:

| Metric | Description |
| --- | --- |
| Answer accuracy | Correct answers on answerable questions |
| Abstention on unanswerable | System correctly refuses when evidence is absent |
| False abstention rate | Answerable questions incorrectly refused |
| Mean confidence | Average retrieval confidence across answered questions |

---

## Demo Questions

```text
"What is the A1C target for most non-pregnant adults with type 2 diabetes?"
"How often should A1C be checked in a stable patient meeting treatment goals?"
"What is the recommended blood pressure target for adults with diabetes?"
"What annual screening is recommended for diabetic kidney disease?"
"Which physical activity recommendations are supported for people with diabetes?"
"Should people taking SGLT2 inhibitors avoid ketogenic diets?"
```

---

## Environment Variables

See `.env.example` for all options. Key ones:

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_MODEL` | `gemma:2b` | Ollama model for generation |
| `GEN_TOPK` | `4` | Chunks passed to LLM context |
| `CONF_ABSTAIN` | `0.35` | Confidence threshold below which the system abstains |
| `SUPPORT_THRESHOLD` | `0.08` | Lexical overlap below which the answer is discarded |
| `FLASK_SECRET_KEY` | random | Flask session signing key |

---

## Limitations & Future Work

- **Table extraction**: ADA numeric targets often appear in tables; PyMuPDF extracts these as plain text which may lose row/column structure. A table-aware extractor (e.g. `camelot`) would improve coverage.
- **Reranker**: A cross-encoder reranker (e.g. `ms-marco-MiniLM`) between retrieval and generation would improve precision without requiring a larger LLM.
- **Streaming UI**: The Flask UI currently waits for the full LLM response; streaming via SSE would improve perceived latency.
- **Multi-turn context**: Session history is stored but not fed back into retrieval; conversational follow-ups lose prior context.

---

## Data Use

This project uses the ADA Standards of Care 2025 for educational and research purposes. For any redistribution or derivative use of the guideline text, follow [ADA's terms](https://diabetesjournals.org).
