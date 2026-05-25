"""
generate.py - Fixed Version
Grounded answerer with improved generation logic
"""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Import your retriever
from retrieve import HybridRetriever

# -------- Config --------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # or llama3.1, mistral, etc.
TOP_K = int(os.getenv("GEN_TOPK", "6"))
CONF_ABSTAIN = float(os.getenv("CONF_ABSTAIN", "0.25"))  # lowered for better coverage
SUPPORT_THRESHOLD = float(os.getenv("SUPPORT_THRESHOLD", "0.10"))  # more lenient

# Patient-friendly question patterns
PATIENT_PATTERNS = {
    'diet': re.compile(r"\b(diet|nutrition|eat|food|snack|drink|beverage|alcohol|juice|sugar|carb|meal)\b", re.I),
    'exercise': re.compile(r"\b(exercise|activity|physical|walk|run|gym|workout|sport)\b", re.I),
    'lifestyle': re.compile(r"\b(lifestyle|weight|sleep|stress|smoking|tobacco)\b", re.I),
    'monitoring': re.compile(r"\b(check|test|monitor|measure|glucose|sugar|a1c)\b", re.I),
    'medication': re.compile(r"\b(medication|medicine|drug|pill|insulin|metformin)\b", re.I),
}

def _has_ollama_installed():
    """Check if ollama is available (either as Python package or CLI)"""
    try:
        import ollama
        return 'python'
    except:
        if shutil.which("ollama"):
            return 'cli'
    return None

def _call_ollama(model: str, messages: List[Dict[str, str]]) -> str:
    """Call Ollama with fallback methods"""
    ollama_type = _has_ollama_installed()
    
    if ollama_type == 'python':
        import ollama
        try:
            resp = ollama.chat(model=model, messages=messages, options={"temperature": 0.3})
            return resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"Ollama Python API failed: {e}")
            return _fallback_generation(messages)
            
    elif ollama_type == 'cli':
        try:
            # Construct prompt for CLI
            system_msg = " ".join([m["content"] for m in messages if m["role"] == "system"])
            user_msg = " ".join([m["content"] for m in messages if m["role"] == "user"])
            prompt = f"{system_msg}\n\n{user_msg}"
            
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            return result.stdout.decode("utf-8").strip()
        except Exception as e:
            print(f"Ollama CLI failed: {e}")
            return _fallback_generation(messages)
    else:
        return _fallback_generation(messages)

def _fallback_generation(messages: List[Dict[str, str]]) -> str:
    """Rule-based fallback when Ollama is unavailable"""
    user_msg = " ".join([m["content"] for m in messages if m["role"] == "user"])
    
    # Extract the excerpts from the user message
    excerpts_match = re.search(r"EXCERPTS:(.*?)FORMAT:", user_msg, re.DOTALL)
    if not excerpts_match:
        return "I don't know based on the loaded ADA 2025 guidelines."
    
    excerpts = excerpts_match.group(1)
    question_match = re.search(r"QUESTION:\s*(.+?)\n", user_msg)
    question = question_match.group(1) if question_match else ""
    
    # Extract relevant sentences based on keywords
    sentences = []
    citations = []
    
    # Parse excerpts
    excerpt_blocks = re.findall(r"\[(\d+)\].*?\n(.+?)(?=\[\d+\]|\Z)", excerpts, re.DOTALL)
    
    for cite_num, text in excerpt_blocks[:3]:  # Use top 3
        # Look for sentences containing key terms from question
        q_terms = set(re.findall(r"\b\w+\b", question.lower()))
        text_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent in text_sentences[:2]:  # Take max 2 sentences per block
            sent_terms = set(re.findall(r"\b\w+\b", sent.lower()))
            if len(q_terms & sent_terms) >= 2:  # At least 2 common terms
                sentences.append(sent.strip())
                if cite_num not in citations:
                    citations.append(cite_num)
                break
    
    if not sentences:
        return "I don't know based on the loaded ADA 2025 guidelines."
    
    # Construct answer
    answer = "Based on the ADA 2025 guidelines, " + " ".join(sentences[:2])
    
    # Add citations
    if citations:
        answer += " [" + "], [".join(citations) + "]"
    
    return answer

def _support_overlap(answer: str, passages: List[str]) -> float:
    """Calculate lexical overlap between answer and source passages"""
    def tokenize(s: str):
        return set(re.findall(r'\b[a-z0-9]+\b', s.lower()))
    
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    
    passage_tokens = set()
    for p in passages:
        passage_tokens |= tokenize(p)
    
    overlap = len(answer_tokens & passage_tokens) / len(answer_tokens)
    return overlap

def _build_prompt(user_query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build a structured prompt for grounded generation"""
    
    # Numbered context blocks
    ctx_lines = []
    for i, d in enumerate(docs, 1):
        header = f"[{i}] Section: {d['section']} (Source: {d['source_id']})"
        ctx_lines.append(header + "\n" + d["text"])
    
    # Detect query type and adjust instructions
    query_type = None
    for qtype, pattern in PATIENT_PATTERNS.items():
        if pattern.search(user_query):
            query_type = qtype
            break
    
    # Base rules
    base_rules = [
        "Answer ONLY using the provided excerpts",
        "If the excerpts don't clearly answer the question, say: 'I don't know based on the loaded ADA 2025 guidelines.'",
        "Use bracketed citations [1], [2] etc. to reference the numbered excerpts",
        "Be concise and precise (3-6 sentences)",
        "Quote specific numbers, targets, or recommendations exactly as written",
        "Never invent information not in the excerpts"
    ]
    
    # Add patient-friendly instructions for lifestyle questions
    if query_type in ['diet', 'exercise', 'lifestyle']:
        base_rules.extend([
            "Use clear, patient-friendly language",
            "Focus on patterns and general guidance rather than absolute restrictions",
            "Mention that recommendations should be individualized when relevant"
        ])
    
    system_prompt = (
        "You are a diabetes care assistant that answers questions strictly based on "
        "the ADA Standards of Care 2025. You have no external knowledge.\n\n"
        "RULES:\n" + "\n".join(f"- {rule}" for rule in base_rules)
    )
    
    user_prompt = (
        f"QUESTION:\n{user_query}\n\n"
        f"EXCERPTS:\n" + "\n\n".join(ctx_lines) + "\n\n"
        "FORMAT:\n"
        "Provide a clear, concise answer in 3-6 sentences with bracketed citations."
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def answer(query: str,
           top_k: int = TOP_K,
           model: str = OLLAMA_MODEL,
           debug: bool = False) -> Dict[str, Any]:
    """Main pipeline: retrieve -> prompt -> generate -> validate"""
    
    # Initialize retriever
    retriever = HybridRetriever(use_reranker=False)
    
    # Retrieve relevant passages
    res = retriever.search(query, k=top_k, strategy="auto", diversify=True, explain=debug)
    
    # Check retriever confidence
    if res.confidence < CONF_ABSTAIN or not res.passages:
        return {
            "query": query,
            "answer": "I don't know based on the loaded ADA 2025 guidelines.",
            "citations": [],
            "confidence": round(res.confidence, 2),
            "used_model": model,
            "support_overlap": 0.0,
            "abstained": True
        }
    
    # Prepare context for generation
    docs = []
    for p in res.passages:
        docs.append({
            "text": p.text,
            "source_id": p.source_id,
            "section": p.section,
            "chunk_idx": p.chunk_idx
        })
    
    # Build prompt
    messages = _build_prompt(query, docs)
    
    # Generate answer
    raw_answer = _call_ollama(model, messages)
    
    # Calculate support overlap
    overlap = _support_overlap(raw_answer, [d["text"] for d in docs])
    
    # Check if answer is valid
    if overlap < SUPPORT_THRESHOLD or "I don't know" in raw_answer:
        return {
            "query": query,
            "answer": "I don't know based on the loaded ADA 2025 guidelines.",
            "citations": [],
            "confidence": round(res.confidence, 2),
            "used_model": model,
            "support_overlap": round(overlap, 3),
            "abstained": True
        }
    
    # Extract citations from answer
    citation_nums = sorted(set(
        int(m) for m in re.findall(r'\[(\d+)\]', raw_answer)
        if 1 <= int(m) <= len(docs)
    ))
    
    # Map citations to source information
    citations = []
    for num in citation_nums:
        doc = docs[num - 1]
        citations.append({
            "number": num,
            "source_id": doc["source_id"],
            "section": doc["section"],
            "chunk_idx": doc["chunk_idx"]
        })
    
    return {
        "query": query,
        "answer": raw_answer.strip(),
        "citations": citations,
        "confidence": round(res.confidence, 2),
        "used_model": model,
        "support_overlap": round(overlap, 3),
        "abstained": False
    }

# -------------- CLI Interface --------------
if __name__ == "__main__":
    import sys
    
    # Default questions for testing
    test_questions = [
        "What is the A1C target for most adults with type 2 diabetes?",
        "What foods should I avoid if I have diabetes?",
        "How much exercise should I get with diabetes?",
        "When should I check my blood sugar?",
        "What medications are used for type 2 diabetes?",
    ]
    
    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("No question provided. Using test question:")
        question = test_questions[0]
        print(f"Q: {question}\n")
    
    # Generate answer
    result = answer(question, top_k=TOP_K, model=OLLAMA_MODEL, debug=False)
    
    # Pretty print result
    print(json.dumps(result, ensure_ascii=False, indent=2))