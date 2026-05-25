"""
run_generic_rag.py
Enhanced script to run the Diabetes Care RAG Assistant using qa.py
"""

import sys
from pathlib import Path

# ------------------------------
# Path Fixes
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# ------------------------------
# CLI functions
# ------------------------------
def test_rag():
    from src.qa import answer as qa_answer
    questions = [
        "What is an appropriate general A1C target for most non-pregnant adults with type 2 diabetes?",
        "How often should an A1C test be performed in a stable patient meeting treatment goals?",
        "What annual screening is recommended for diabetic kidney disease?",
        "What is the recommended blood pressure target for most adults with diabetes?",
        "What lifestyle recommendations are supported for physical activity?",
    ]
    for q in questions:
        print("\n❓", q)
        res = qa_answer(q)
        print("💬", res.answer)
        print("📚 Sources:", res.citations)


def answer_single_question(question: str):
    from src.qa import answer as qa_answer
    print(f"\n❓ Question: {question}")
    res = qa_answer(question)
    print(f"\n💬 Answer:\n{res.answer}")
    print("\n📚 Sources:")
    for c in res.citations:
        print(f" - {c['section']} ({c['source_id']}, chunk {c['chunk_idx']})")


# ------------------------------
# Main entry point
# ------------------------------
def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_rag()
        elif sys.argv[1] == '--web':
            from src.web_ui import run_web_ui
            run_web_ui()
        elif sys.argv[1] == '--evaluate':
            from src.evaluate_gold import main as run_eval
            run_eval()
        else:
            q = " ".join(sys.argv[1:])
            answer_single_question(q)
    else:
        test_rag()


if __name__ == "__main__":
    main()
