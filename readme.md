# Diabetes Clinical Q&A (RAG) — ADA 2025

Evidence-based Q&A grounded in ADA Standards of Care in Diabetes (2025) using a hybrid retriever (BM25 + embeddings + fusion). The system answers from your local guideline corpus and abstains if the evidence isn’t present—no web access, no hallucinations by design.

✨ What’s in this repo

Runtime (for the demo):

src/run_generic_rag.py — CLI entrypoint (test suite, single Q&A, evaluation)

src/generic_rag.py — end-to-end RAG pipeline (uses built indexes)

src/retrieve.py — hybrid retrieval (BM25 + dense + fusion)

src/evaluate_gold.py — evaluation on your gold set

Build pipeline (only if you rebuild the index):

src/ingest.py — extract ADA PDF → Markdown sections

src/chunk.py — split Markdown into overlapping chunks

src/bm25.py — build BM25 index

src/embed_index.py — build embeddings + Annoy index

(Optional)

src/presets.py — common questions for a UI/testing

📁 Expected layout
diabetes-rag/
  data/
    raw/                       # (optional) ADA PDF here
    clean/                     # Markdown sections (one per ADA chapter)
    eval/
      gold_diabetes_80.json    # gold evaluation set (if provided)
  index/
    chunks.jsonl
    meta.jsonl
    bm25.pkl
    bm25_meta.jsonl
    embeddings.npy
    annoy_cosine.idx
  src/
    __init__.py
    run_generic_rag.py
    generic_rag.py
    retrieve.py
    evaluate_gold.py
    ingest.py
    chunk.py
    bm25.py
    embed_index.py
    presets.py


If index/ already exists with those files, you can run the system immediately—no rebuild needed.

🧰 Prerequisites

Python 3.10–3.12

(Optional) Ollama if you plan to generate natural-language answers locally

Install: https://ollama.com/download

Pull a model (default used in code): ollama pull llama3.1

Create & activate a virtual env
# Windows (PowerShell)
py -3.11 -m venv .venv
. .venv/Scripts/Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

Install Python deps

If you have requirements.txt:

pip install -r requirements.txt


Minimal set (works for this repo):

pip install numpy tqdm regex annoy rank-bm25 sentence-transformers
# Optional for local generation:
pip install ollama
# If torch is missing on CPU:
pip install --index-url https://download.pytorch.org/whl/cpu torch

🚀 Quick start (no rebuilding)

From the repo root:

1) Mini test suite
python src/run_generic_rag.py --test

2) Ask a single question
python src/run_generic_rag.py "Which conditions can make A1C results unreliable?"

3) Evaluate on your gold set
python src/run_generic_rag.py --evaluate


By design, if a specific answer is not clearly present in your corpus, the system abstains rather than inventing one.

🛠️ (Optional) Rebuild the corpus & indexes

Only needed if you’ve changed the PDF or added/edited Markdown.

Extract ADA PDF → Markdown

Put the PDF at data/raw/standards-of-care-2025.pdf (or adjust the path in ingest.py)

python src/ingest.py


Expect ~15–17 .md files in data/clean/.

Chunk the Markdown

python src/chunk.py


Build BM25

python src/bm25.py


Build embeddings + Annoy

python src/embed_index.py


Now re-run the quick start commands.

💬 Good demo questions (work well with the current corpus)

Monitoring & A1C

“When should A1C be checked more frequently than usual in diabetes?”

“Which conditions can make A1C results unreliable?”

“Are NGSP-certified A1C assays recommended for routine testing?”

Therapy & safety

“Should people taking SGLT2 inhibitors avoid ketogenic diets?”

“What patient education reduces ketoacidosis risk with SGLT2 inhibitors?”

Screening & general care

“What annual labs are recommended to monitor diabetic kidney disease?”

“How often should feet be examined in people with diabetes?”

Lifestyle & activity

“Which eating patterns align with cardiometabolic health in diabetes?”

“What physical activity is recommended for people with diabetes?”

⚙️ Environment variables (optional)
# change top-k passages for generation (if enabled)
export GEN_TOPK=6          # (Windows PowerShell: setx GEN_TOPK 6)

# pick an Ollama model
export OLLAMA_MODEL=llama3.1

# retrieval confidence below which we abstain
export CONF_ABSTAIN=0.35

🧪 What to show in your slides

Architecture: Ingest → Chunk → Hybrid Retrieval → (optional) Local Generator → Answer+citations/abstain

Demo: 3–5 CLI screenshots with answers and cited source lines

Metrics: Output from --evaluate (accuracy, abstain rate)

Safety: Emphasize abstention over hallucination

Future work: table-aware extraction, section pinning for numeric targets, optional reranker

❓ Troubleshooting

ModuleNotFoundError: retrieve
Run from the repo root; ensure src/__init__.py exists (even if empty).

Torch/TorchVision mismatch warnings
For CPU-only usage, it’s fine to remove torchvision:

pip uninstall torchvision -y


Frequent abstains on numeric targets
Those targets may live in tables/figures that didn’t extract. Check the corresponding Markdown in data/clean/, then re-run: chunk.py → bm25.py → embed_index.py.

📄 License / data use

This project uses the ADA Standards of Care for educational purposes. Follow ADA’s terms for any redistribution or derivative use of the guideline text.