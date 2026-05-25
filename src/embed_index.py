# src/embed_index.py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

BASE_DIR = Path(__file__).resolve().parent.parent
IDX_DIR = BASE_DIR / "index"
IDX_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_JSONL = IDX_DIR / "chunks.jsonl"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # fast & good
EMB_NPY = IDX_DIR / "embeddings.npy"
META_JSONL = IDX_DIR / "meta.jsonl"
ANNOY_IDX = IDX_DIR / "annoy_cosine.idx"

def load_chunks():
    chunks = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def build_embeddings(texts):
    model = SentenceTransformer(EMB_MODEL, trust_remote_code=True)
    # Return L2-normalized vectors (cosine = dot)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embs, dtype=np.float32)

def build_annoy(embs: np.ndarray, n_trees=50):
    dim = embs.shape[1]
    index = AnnoyIndex(dim, metric="angular")  # angular ≈ cosine
    for i in tqdm(range(len(embs)), desc="Annoy add"):
        index.add_item(i, embs[i])
    index.build(n_trees)
    index.save(ANNOY_IDX.as_posix())

def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks")

    embs = build_embeddings(texts)
    np.save(EMB_NPY, embs)
    print(f"✅ Saved embeddings → {EMB_NPY} shape={embs.shape}")

    # Save metadata in same order as embeddings
    with META_JSONL.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ Saved metadata → {META_JSONL}")

    build_annoy(embs, n_trees=50)
    print(f"✅ Saved Annoy index → {ANNOY_IDX}")

if __name__ == "__main__":
    main()
