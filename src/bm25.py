# src/bm25.py
import re, json, pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

CHUNKS_JSONL = Path("index/chunks.jsonl")
IDX_DIR = Path("index")
IDX_DIR.mkdir(parents=True, exist_ok=True)


BM25_PKL = IDX_DIR / "bm25.pkl"
BM25_META = IDX_DIR / "bm25_meta.jsonl"

def tok(s: str):
    return re.findall(r"\w+(?:'\w+)?", s.lower(), flags=re.UNICODE)

def load_chunks():
    data = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    chunks = load_chunks()
    corpus_tokens = [tok(c["text"]) for c in tqdm(chunks, desc="Tokenizing")]
    bm25 = BM25Okapi(corpus_tokens)
    with open(BM25_PKL, "wb") as f:
        pickle.dump({"bm25": bm25}, f)
    print(f"✅ Saved BM25 → {BM25_PKL}")

    with BM25_META.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ Saved BM25 meta → {BM25_META}")

if __name__ == "__main__":
    main()
