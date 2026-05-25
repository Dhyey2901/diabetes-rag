"""
generic_rag.py
Generic RAG system that automatically adapts to any corpus without hardcoding
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from collections import Counter

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generic_rag")

# ------------------------------
# Path Fixes (always relative to project root)
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # parent of src
INDEX_DIR = BASE_DIR / "index"

CHUNKS_PATH     = INDEX_DIR / "chunks.jsonl"
BM25_PATH       = INDEX_DIR / "bm25.pkl"
META_PATH       = INDEX_DIR / "meta.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
ANNOY_PATH      = INDEX_DIR / "annoy_cosine.idx"

def _check_file(path: Path):
    """Ensure a file exists, otherwise raise with clear log."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Required file missing: {path}")
    logger.info(f"✅ Found: {path}")
    return path

# ============= Automatic Corpus Analysis =============
class CorpusAnalyzer:
    """Automatically analyze corpus to understand its content and structure"""
    
    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks_path = _check_file(chunks_path)
        self.chunks = self._load_chunks(self.chunks_path)
        self.corpus_stats = self._analyze_corpus()
        self.domain_terms = self._extract_domain_terms()
        self.topic_clusters = self._identify_topics()
    
    def _load_chunks(self, path: Path) -> List[Dict]:
        """Load chunks from JSONL"""
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
    
    def _analyze_corpus(self) -> Dict[str, Any]:
        stats = {
            'total_chunks': len(self.chunks),
            'sources': list(set(c.get('source_id', '') for c in self.chunks)),
            'sections': list(set(c.get('section', '') for c in self.chunks)),
            'avg_chunk_length': np.mean([len(c.get('text', '').split()) for c in self.chunks]),
        }
        return stats
    
    def _extract_domain_terms(self) -> Dict[str, float]:
        term_freq = Counter()
        doc_freq = Counter()
        
        for chunk in self.chunks:
            text = chunk.get('text', '').lower()
            terms = set(re.findall(r'\b[a-z]+(?:[-][a-z]+)*\b', text))
            
            for term in terms:
                doc_freq[term] += 1
            
            all_terms = re.findall(r'\b[a-z]+(?:[-][a-z]+)*\b', text)
            term_freq.update(all_terms)
        
        domain_terms = {}
        total_chunks = len(self.chunks)
        
        for term, tf in term_freq.most_common(500):
            if len(term) > 2 and doc_freq[term] > 1:
                idf = np.log(total_chunks / (1 + doc_freq[term]))
                score = tf * idf
                if score > 10:
                    domain_terms[term] = score
        
        return domain_terms
    
    def _identify_topics(self) -> Dict[str, List[str]]:
        topics = {}
        for chunk in self.chunks:
            section = chunk.get('section', 'general')
            if section not in topics:
                topics[section] = []
            text = chunk.get('text', '').lower()
            key_terms = [term for term in self.domain_terms.keys() if term in text][:10]
            topics[section].extend(key_terms)
        
        for section in topics:
            term_counts = Counter(topics[section])
            topics[section] = [term for term, _ in term_counts.most_common(20)]
        
        return topics

# ============= Intelligent Query Processor =============
class QueryProcessor:
    def __init__(self, corpus_analyzer: CorpusAnalyzer):
        self.analyzer = corpus_analyzer
        self.domain_terms = corpus_analyzer.domain_terms
    
    def process_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        query_type = self._classify_query_type(query)
        relevant_terms = [term for term in self.domain_terms.keys() if term in query_lower]
        best_topic = self._find_best_topic(query_lower)
        enhancements = self._generate_enhancements(query_lower, relevant_terms)
        
        return {
            'original': query,
            'type': query_type,
            'domain_terms': relevant_terms,
            'topic': best_topic,
            'enhancements': enhancements,
            'enhanced_query': self._build_enhanced_query(query, enhancements)
        }
    
    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()
        patterns = {
            'definition': r'\b(what is|what are|define|meaning of)\b',
            'procedure': r'\b(how to|how do|how should|steps to)\b',
            'comparison': r'\b(difference|compare|versus|vs|better)\b',
            'recommendation': r'\b(should|recommend|best|optimal|appropriate)\b',
            'frequency': r'\b(how often|how many times|frequency|when to)\b',
            'criteria': r'\b(when|criteria|indication|qualify)\b',
            'list': r'\b(list|examples|types of|kinds of)\b',
            'explanation': r'\b(why|explain|reason|cause)\b',
        }
        for qtype, pattern in patterns.items():
            if re.search(pattern, query_lower):
                return qtype
        return 'general'
    
    def _find_best_topic(self, query: str) -> Optional[str]:
        best_topic = None
        best_score = 0
        for topic, terms in self.analyzer.topic_clusters.items():
            score = sum(1 for term in terms if term in query)
            if score > best_score:
                best_score = score
                best_topic = topic
        return best_topic if best_score > 0 else 'general'
    
    def _generate_enhancements(self, query: str, relevant_terms: List[str]) -> List[str]:
        enhancements = []
        for term in relevant_terms[:3]:
            related = self._find_related_terms(term)
            enhancements.extend(related[:2])
        return list(set(enhancements))
    
    def _find_related_terms(self, term: str) -> List[str]:
        related = []
        for chunk in self.analyzer.chunks:
            if term in chunk.get('text', '').lower():
                text = chunk.get('text', '').lower()
                for other_term in self.domain_terms.keys():
                    if other_term != term and other_term in text:
                        related.append(other_term)
        term_counts = Counter(related)
        return [t for t, _ in term_counts.most_common(3)]
    
    def _build_enhanced_query(self, original: str, enhancements: List[str]) -> str:
        if enhancements:
            return f"{original} ({' '.join(enhancements[:3])})"
        return original

# ============= Adaptive Retriever =============
class AdaptiveRetriever:
    def __init__(self, corpus_analyzer: CorpusAnalyzer):
        self.analyzer = corpus_analyzer
        self.query_processor = QueryProcessor(corpus_analyzer)
        self._load_indices()
    
    def _load_indices(self):
        from retrieve import HybridRetriever
        # Ensure required index files exist
        _check_file(META_PATH)
        _check_file(EMBEDDINGS_PATH)
        _check_file(ANNOY_PATH)
        _check_file(BM25_PATH)
        self.hybrid_retriever = HybridRetriever(use_reranker=False)
    
    def retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        query_info = self.query_processor.process_query(query)
        enhanced_query = query_info['enhanced_query']
        results = self.hybrid_retriever.search(enhanced_query, k=top_k, strategy="auto", diversify=True)
        if not results or not results.passages:
            return []
        chunks = []
        for passage in results.passages:
            chunks.append({
                'text': passage.text,
                'source_id': passage.source_id,
                'section': passage.section,
                'chunk_idx': passage.chunk_idx,
                'score': passage.score,
                'query_type': query_info['type'],
                'topic_match': passage.section == query_info['topic']
            })
        return self._adaptive_rerank(chunks, query_info)
    
    def _adaptive_rerank(self, chunks: List[Dict], query_info: Dict) -> List[Dict]:
        for chunk in chunks:
            if chunk.get('topic_match'):
                chunk['score'] *= 1.2
        chunks.sort(key=lambda x: x['score'], reverse=True)
        return chunks

# ============= Smart Answer Generator =============
class SmartAnswerGenerator:
    def __init__(self, corpus_analyzer: CorpusAnalyzer):
        self.analyzer = corpus_analyzer
    
    def generate_answer(self, query: str, chunks: List[Dict], query_info: Dict) -> Dict[str, Any]:
        if not chunks:
            return {
                'query': query,
                'answer': "I don't have sufficient information.",
                'confidence': 0.0,
                'query_type': query_info['type'],
                'chunks_used': [],
                'abstained': True
            }
        facts = [c['text'] for c in chunks[:2]]
        answer = " ".join(facts)[:500]
        return {
            'query': query,
            'answer': answer,
            'confidence': 0.75,
            'query_type': query_info['type'],
            'chunks_used': [c['chunk_idx'] for c in chunks[:2]],
            'abstained': False
        }

# ============= Main RAG Pipeline =============
class GenericRAG:
    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        logger.info("Initializing Generic RAG System...")
        self.corpus_analyzer = CorpusAnalyzer(chunks_path)
        self.retriever = AdaptiveRetriever(self.corpus_analyzer)
        self.generator = SmartAnswerGenerator(self.corpus_analyzer)
    
    def answer(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        query_info = self.retriever.query_processor.process_query(query)
        chunks = self.retriever.retrieve(query, top_k)
        return self.generator.generate_answer(query, chunks, query_info)
    
    def get_corpus_info(self) -> Dict[str, Any]:
        return {
            'stats': self.corpus_analyzer.corpus_stats,
            'topics': list(self.corpus_analyzer.topic_clusters.keys()),
            'top_domain_terms': list(self.corpus_analyzer.domain_terms.keys())[:20]
        }

# ============= CLI =============
if __name__ == "__main__":
    import sys
    rag = GenericRAG()
    print("\n📚 Corpus Information:")
    info = rag.get_corpus_info()
    print(f"  • Total chunks: {info['stats']['total_chunks']}")
    print(f"  • Topics: {', '.join(info['topics'][:5])}")
    print(f"  • Domain terms: {', '.join(info['top_domain_terms'][:10])}")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the A1C target for adults with diabetes?"
    print(f"\n❓ Query: {query}")
    response = rag.answer(query)
    print(f"\nQuery Type: {response.get('query_type', 'unknown')}")
    print(f"Confidence: {response.get('confidence', 0):.2%}")
    print(f"Abstained: {response.get('abstained', False)}")
    print(f"\n💬 Answer:\n{response['answer']}")
