# src/qa.py
"""
User-ready QA pipeline (Ollama version with HTTP API + Streaming):
- Two-pass retrieval with auto-detected section pinning
- Smart context reducer
- Strict, citation-first prompt with clean abstention
- Streaming output (fixed JSON parsing)
"""

from __future__ import annotations
import os, re, json, requests
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from .retrieve import HybridRetriever

# ---------- Config ----------
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # default to mistral (fast + lightweight)
TOP_K = int(os.getenv("GEN_TOPK", "4"))
CONF_ABSTAIN = float(os.getenv("CONF_ABSTAIN", "0.35"))
SUPPORT_THRESHOLD = float(os.getenv("SUPPORT_THRESHOLD", "0.08"))

# intents
RX_A1C    = re.compile(r"\b(a1c|hb?a1c|glycemic\s+(goal|target))\b", re.I)
RX_DIET   = re.compile(r"\b(diet|nutrition|eat|foods?|snack|drink|beverage|soda|juice|alcohol)\b", re.I)
RX_KIDNEY = re.compile(r"\b(ckd|kidney|egfr|uacr|albumin)\b", re.I)

# ---------- Ollama wrapper with streaming ----------
def _ollama_chat(model: str, messages: List[Dict[str, str]]) -> str:
    try:
        with requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "options": {"temperature": 0.2, "num_predict": 300}
            },
            stream=True
        ) as r:
            r.raise_for_status()
            collected = []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                if "message" in data and "content" in data["message"]:
                    collected.append(data["message"]["content"])

                if data.get("done", False):
                    break

            return "".join(collected).strip()
    except Exception as e:
        return f"[Error calling Ollama API: {e}]"

# ---------- Lexical + context helpers ----------
def _lexical_support(answer: str, passages: List[str]) -> float:
    def toks(s: str): return set(re.findall(r"[A-Za-z0-9%\.]+", s.lower()))
    a = toks(answer)
    if not a: return 0.0
    U = set()
    for p in passages: U |= toks(p)
    return len(a & U) / max(1, len(a))

def _reduce_context(passages: List[Dict[str, Any]], query: str, keep: int = 6) -> List[Dict[str, Any]]:
    q_terms = re.findall(r"[A-Za-z0-9%\.]+", query.lower())
    def jaccard(a: set, b: set): 
        return len(a & b)/len(a | b) if a and b else 0.0

    RX_NUM = re.compile(r"\b\d+(\.\d+)?\s*(%|mmhg|mg/dl|mmol/l|years?|months?|weeks?)\b", re.I)
    RX_TARGET = re.compile(r"\b(target|goal|recommended)\b", re.I)

    scored = []
    for p in passages:
        toks = set(re.findall(r"[A-Za-z0-9%\.]+", p["text"].lower()))
        cov = jaccard(set(q_terms), toks)
        has_num = 0.12 if RX_NUM.search(p["text"]) else 0.0
        has_target_word = 0.08 if RX_TARGET.search(p["text"]) else 0.0
        sec_bonus = 0.0
        sec = p.get("section","").lower()
        if RX_A1C.search(query) and "glycemic" in sec: sec_bonus += 0.20
        if RX_DIET.search(query) and any(x in sec for x in ["behavior","behaviors","lifestyle","obesity","weight"]): sec_bonus += 0.15
        if RX_KIDNEY.search(query) and "kidney" in sec: sec_bonus += 0.15
        score = cov + has_num + has_target_word + sec_bonus
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    chosen, seen_keys = [], set()
    for s,p in scored:
        key = (p["source_id"], p["chunk_idx"]//2)
        if key in seen_keys: 
            continue
        chosen.append(p)
        seen_keys.add(key)
        if len(chosen) >= keep:
            break
    return chosen

def _build_prompt(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    ctx_lines = []
    for i, d in enumerate(docs, 1):
        header = f"[{i}] {d['section']} — {d['source_id']} (chunk {d['chunk_idx']})"
        ctx_lines.append(header + "\n" + d["text"])

    rules = [
        "Answer ONLY using the provided excerpts. If an answer is not clearly supported, reply: I don’t know based on the loaded ADA 2025 guidelines.",
        "Add bracketed citation numbers [1], [2] after the sentences they support.",
        "Be concise (3–6 sentences), precise, and avoid speculation.",
        "If targets, thresholds, or frequencies are present, quote them exactly.",
        "Do not invent numbers or sources."
    ]
    if RX_DIET.search(query):
        rules += [
            "Use clear, patient-friendly language.",
            "Prefer pattern-level advice unless excerpts explicitly ban a specific food.",
            "Add one short sentence: advice should be individualized."
        ]

    sys = "You are an ADA 2025 guideline-grounded assistant. You have NO external access.\n" + \
          "RULES:\n- " + "\n- ".join(rules)
    user = (
        f"QUESTION:\n{query}\n\nEXCERPTS:\n" + "\n\n".join(ctx_lines) + "\n\n"
        "FORMAT:\n"
        "• Write the answer in 3–6 sentences.\n"
        "• Include bracketed citations [n] at sentence ends.\n"
        "• If unknown, respond exactly: I don’t know based on the loaded ADA 2025 guidelines."
    )
    return [{"role":"system","content":sys},{"role":"user","content":user}]

# ---------- Public API ----------
@dataclass
class QAResult:
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    support_overlap: float
    used_model: str

def answer(query: str, top_k: int = TOP_K, model: str = OLLAMA_MODEL) -> QAResult:
    retriever = HybridRetriever(use_reranker=False)
    passages_all = retriever.search(query, k=max(10, top_k*2))
    conf = passages_all.confidence

    passages = [{
        "text": p.text, "source_id": p.source_id, "section": p.section, "chunk_idx": p.chunk_idx
    } for p in passages_all.passages]

    if conf < CONF_ABSTAIN or not passages:
        return QAResult(query, "I don’t know based on the loaded ADA 2025 guidelines.", [], round(conf,2), 0.0, model)

    pack = _reduce_context(passages, query, keep=top_k)
    msgs = _build_prompt(query, pack)
    raw = _ollama_chat(model, msgs)

    overlap = _lexical_support(raw, [p["text"] for p in pack])
    if overlap < SUPPORT_THRESHOLD or not raw.strip():
        return QAResult(query, "I don’t know based on the loaded ADA 2025 guidelines.", [], round(conf,2), round(overlap,3), model)

    nums = sorted({int(x) for x in re.findall(r"\[(\d{1,2})\]", raw) if 1 <= int(x) <= len(pack)})
    mapped = [{"n": n, "source_id": pack[n-1]["source_id"], "section": pack[n-1]["section"], "chunk_idx": pack[n-1]["chunk_idx"]} for n in nums]

    return QAResult(query, raw.strip(), mapped, round(conf,2), round(overlap,3), model)

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "A1c target for nonpregnant adults with type 2 diabetes"
    res = answer(q)
    print("\n--- FINAL RESULT ---")
    print(json.dumps({
        "query": res.query,
        "answer": res.answer,
        "citations": res.citations,
        "confidence": res.confidence,
        "support_overlap": res.support_overlap,
        "used_model": res.used_model
    }, ensure_ascii=False, indent=2))
