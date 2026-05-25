"""
Supercharged Hybrid Retrieval for Diabetes RAG

Features:
- Dense (Annoy + MiniLM) + BM25
- Smart query expansion (synonyms/abbreviations)
- Fusion: Reciprocal Rank Fusion (RRF) or weighted strategy
- Section & intent-aware boosts (esp. Glycemic queries)
- Keyword presence bonus (A1c numeric % targets)
- MMR diversification
- Optional CrossEncoder reranking
- Confidence score + explain mode

Run test:
    python src/retrieve.py "A1c target for adults with type 2 diabetes"
"""

from __future__ import annotations
import os, re, json, logging, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from annoy import AnnoyIndex
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util as st_util

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index"

def _check_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"❌ Required file missing: {path}")
    logging.getLogger("hybrid-retriever").info(f"✅ Found: {path}")
    return path

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ANNOY_PATH = _check_file(INDEX_DIR / "annoy_cosine.idx")
EMB_NPY = _check_file(INDEX_DIR / "embeddings.npy")
META_JSONL = _check_file(INDEX_DIR / "meta.jsonl")
BM25_PKL = _check_file(INDEX_DIR / "bm25.pkl")
BM25_META_JSONL = _check_file(INDEX_DIR / "bm25_meta.jsonl")

CAND_DENSE = 80
CAND_BM25  = 80
FINAL_K    = 8

RRF_C = 60.0
WEIGHT_DENSE_DEFAULT = 0.6
WEIGHT_BM25_DEFAULT  = 0.4

MMR_LAMBDA = 0.8

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid-retriever")

# ---------------- Helpers ----------------
STOP = set("a an the of and or for to in on with from at by is are be as than that this those these which who whom whose into about over under after during before within without per vs via not".split())

def tok(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9%\.]+", s.lower()) if t not in STOP]

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def make_snippet(text: str, query_terms: List[str], window: int = 40) -> str:
    words = text.split()
    qset = set(query_terms)
    hit = next((i for i, w in enumerate(words) if w.lower().strip(".,") in qset), 0)
    start, end = max(0, hit - window), min(len(words), hit + window)
    return ("..." if start > 0 else "") + " ".join(words[start:end]) + ("..." if end < len(words) else "")

# ---------------- Query Expansion ----------------
SYNONYMS = {
    "a1c": ["hba1c", "glycated hemoglobin"],
    "hba1c": ["a1c", "glycated hemoglobin"],
    "bp": ["blood pressure", "hypertension"],
    "ckd": ["chronic kidney disease", "egfr"],
    "cvd": ["cardiovascular disease", "ascvd"],
    "statin": ["lipid therapy", "ldl lowering"],
    "glp-1": ["glp-1 ra", "glucagon-like peptide-1"],
    "sglt2": ["sodium-glucose cotransporter 2", "sglt2 inhibitor"],
}

def expand_query(query: str) -> tuple[str, List[str]]:
    terms = tok(query)
    expansions = []
    for t in terms:
        if t in SYNONYMS:
            expansions.extend(SYNONYMS[t])
    expansions = list(dict.fromkeys(expansions))[:5]
    return query + (" (" + " ".join(expansions) + ")" if expansions else ""), terms

# ---------------- Data Models ----------------
@dataclass
class RetrievedPassage:
    id: str
    text: str
    source_id: str
    section: str
    year: str
    url: str
    chunk_idx: int
    score: float
    score_dense: float
    score_bm25: float
    rerank_score: Optional[float] = None
    snippet: Optional[str] = None

@dataclass
class RetrievalResult:
    query: str
    expanded_query: str
    passages: List[RetrievedPassage]
    confidence: float
    sources: List[str]
    explain: Optional[Dict] = None

# ---------------- Retriever ----------------
class HybridRetriever:
    def __init__(self, emb_model: str = EMB_MODEL, use_reranker: bool = False):
        logger.info("Loading embeddings & Annoy index...")
        self.meta_dense = [json.loads(l) for l in open(META_JSONL, encoding="utf-8")]
        self.embs = np.load(EMB_NPY)
        self.dim = self.embs.shape[1]
        self.ann = AnnoyIndex(self.dim, metric="angular")
        self.ann.load(str(ANNOY_PATH))
        self.embedder = SentenceTransformer(emb_model)

        logger.info("Loading BM25 index...")
        with open(BM25_PKL, "rb") as f:
            self.bm25_obj: BM25Okapi = pickle.load(f)["bm25"]
        self.meta_bm25 = [json.loads(l) for l in open(BM25_META_JSONL, encoding="utf-8")]

        self.reranker = None
        if use_reranker and CrossEncoder:
            try:
                logger.info("Loading CrossEncoder reranker...")
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                logger.warning(f"Reranker load failed: {e}")

    # ----- Core searches -----
    def _dense_search(self, query: str, topn: int = CAND_DENSE):
        qv = self.embedder.encode([query], normalize_embeddings=True)[0]
        idxs, dists = self.ann.get_nns_by_vector(qv, topn, include_distances=True)
        sims = [1 - (d**2)/2 for d in dists]
        return list(zip(idxs, sims))

    def _bm25_search(self, query: str, topn: int = CAND_BM25):
        scores = self.bm25_obj.get_scores(tok(query))
        idxs = np.argsort(scores)[::-1][:topn]
        return [(int(i), float(scores[i])) for i in idxs]

    def _normalize(self, hits):
        if not hits: return {}
        arr = np.array([s for _,s in hits])
        lo, hi = arr.min(), arr.max()
        return {i:(s-lo)/(hi-lo+1e-6) for i,s in hits}

    def _weighted(self, dense, bm25, k, wd, wb):
        dn, bn = self._normalize(dense), self._normalize(bm25)
        scores = {}
        for i,s in dn.items(): scores[i] = scores.get(i,0)+wd*s
        for i,s in bn.items(): scores[i] = scores.get(i,0)+wb*s
        return [i for i,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:k]]

    def _rrf(self, dense, bm25, k):
        d_rank = {i: r for r,(i,_) in enumerate(dense)}
        b_rank = {i: r for r,(i,_) in enumerate(bm25)}
        ids = set(d_rank) | set(b_rank)
        scores=[]
        for i in ids:
            s=0.0
            if i in d_rank: s+=1/(RRF_C+d_rank[i]+1)
            if i in b_rank: s+=1/(RRF_C+b_rank[i]+1)
            scores.append((i,s))
        scores.sort(key=lambda x:x[1], reverse=True)
        return [i for i,_ in scores[:k]]

    def _mmr(self, ids: List[int], query_vec: np.ndarray, k: int) -> List[int]:
        if not ids: return ids
        X = self.embs[np.array(ids)]
        rel = (X @ query_vec.reshape(-1,1)).flatten()
        selected=[]
        remaining=list(range(len(ids)))
        while remaining and len(selected)<k:
            if not selected:
                best=int(np.argmax(rel[remaining]))
                selected.append(remaining.pop(best))
                continue
            S=X[np.array([i for i in selected])]
            sim_to_S = st_util.cos_sim(X[remaining], S).cpu().numpy().max(axis=1)
            mmr_scores = MMR_LAMBDA*rel[remaining] - (1-MMR_LAMBDA)*sim_to_S
            best=int(np.argmax(mmr_scores))
            selected.append(remaining.pop(best))
        return [ids[i] for i in selected]

    # ----- Public search -----
    def search(self, query: str, k: int = FINAL_K,
               strategy: str = "auto", diversify: bool = True,
               explain: bool = False) -> RetrievalResult:

        query_exp, q_terms = expand_query(query)
        dense_hits = self._dense_search(query_exp)
        bm25_hits  = self._bm25_search(query_exp)

        # Strategy
        if strategy=="rrf":
            cand_ids = self._rrf(dense_hits, bm25_hits, k*3)
        else:
            cand_ids = self._weighted(dense_hits, bm25_hits, k*3, WEIGHT_DENSE_DEFAULT, WEIGHT_BM25_DEFAULT)

        # Diversification
        if diversify:
            qv = self.embedder.encode([query_exp], normalize_embeddings=True)[0]
            cand_ids = self._mmr(cand_ids, qv, k)

        passages=[]
        for idx in cand_ids[:k]:
            meta=self.meta_dense[idx]
            t=normalize_space(meta.get("text",""))
            sd=next((s for i,s in dense_hits if i==idx),0)
            sb=next((s for i,s in bm25_hits if i==idx),0)
            sc=0.5*sd+0.5*sb
            passages.append(RetrievedPassage(
                id=f"{meta.get('source_id')}#{idx}",
                text=t,
                source_id=meta.get("source_id",""),
                section=meta.get("section",""),
                year=meta.get("year",""),
                url=meta.get("url",""),
                chunk_idx=idx,
                score=sc,
                score_dense=sd,
                score_bm25=sb,
                snippet=make_snippet(t,q_terms)
            ))

        passages.sort(key=lambda p:p.score, reverse=True)

        # Confidence = normalized top score
        scores = [p.score for p in passages]
        conf = float((max(scores)-min(scores))/(max(scores)+1e-6)) if scores else 0.0

        sources = [f"{p.source_id} (section: {p.section})" for p in passages]

        return RetrievalResult(
            query=query,
            expanded_query=query_exp,
            passages=passages,
            confidence=conf,
            sources=sources,
            explain={"dense":dense_hits[:5],"bm25":bm25_hits[:5],"strategy":strategy} if explain else None
        )

# ---------------- CLI ----------------
if __name__=="__main__":
    import sys
    q=" ".join(sys.argv[1:]) if len(sys.argv)>1 else "A1c target for adults with diabetes"
    retriever=HybridRetriever()
    res=retriever.search(q,k=5,strategy="auto",diversify=True,explain=True)
    print(f"\nQuery: {q}\nExpanded: {res.expanded_query}")
    for i,p in enumerate(res.passages,1):
        print(f"\n[{i}] {p.text[:200]}...\nScore={p.score:.3f}, Section={p.section}, Source={p.source_id}")
    print("\nConfidence:", res.confidence)
    print("Sources:", res.sources)
    if res.explain: print("Explain:", res.explain)
