# src/chunk.py
import re, json, math
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

DATA_DIR = Path("data/clean")
OUT_JSONL = Path("index/chunks.jsonl")
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# Simple word tokenizer
def _words(txt: str) -> List[str]:
    return re.findall(r"\w+(?:'\w+)?", txt, flags=re.UNICODE)

def _join(words: List[str]) -> str:
    return " ".join(words)

def chunk_text(text: str, target_words=500, overlap_words=100) -> List[str]:
    ws = _words(text)
    if not ws:
        return []
    chunks = []
    step = max(1, target_words - overlap_words)
    for start in range(0, len(ws), step):
        end = min(len(ws), start + target_words)
        chunk = _join(ws[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(ws):
            break
    return chunks

def guess_section_title(md_path: Path) -> str:
    # Try from first markdown heading; fallback to filename
    txt = md_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^\s{0,3}#{1,3}\s+(.+)$", txt, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback: filename prettified
    return md_path.stem.replace("_", " ")

def build_chunks():
    files = sorted(DATA_DIR.glob("ADA2025_*.md"))
    items: List[Dict] = []
    gid = 0
    for fp in tqdm(files, desc="Chunking"):
        section_title = guess_section_title(fp)
        source_id = fp.stem  # e.g., ADA2025_06_Glycemic_Goals_and_Hypoglycemia
        text = fp.read_text(encoding="utf-8", errors="ignore")
        # Strip very long whitespace runs
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        chunks = chunk_text(text, target_words=500, overlap_words=100)
        for i, ch in enumerate(chunks):
            items.append({
                "id": f"{source_id}__{i}",
                "text": ch,
                "source_id": source_id,
                "section": section_title,
                "year": "2025",
                "url": "",             # (optional) add if you want live links in UI
                "chunk_idx": i,
            })
            gid += 1

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(items)} chunks → {OUT_JSONL}")

if __name__ == "__main__":
    build_chunks()
