# src/embed_index.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
IDX_DIR = BASE_DIR / "index"
IDX_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_JSONL = IDX_DIR / "chunks.jsonl"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_NPY   = IDX_DIR / "embeddings.npy"
META_JSONL = IDX_DIR / "meta.jsonl"

def load_chunks():
    chunks = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def build_embeddings(texts):
    model = SentenceTransformer(EMB_MODEL, trust_remote_code=True)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embs, dtype=np.float32)

def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks")

    embs = build_embeddings(texts)
    np.save(EMB_NPY, embs)
    print(f"Saved embeddings → {EMB_NPY} shape={embs.shape}")

    with META_JSONL.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved metadata → {META_JSONL}")

if __name__ == "__main__":
    main()
