"""
evaluate_gold.py
Evaluate the QA pipeline (qa.py) using the gold standard diabetes dataset.
Run from project root: python src/evaluate_gold.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Ensure project root is on the path so src.qa relative imports resolve
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.qa import answer as qa_answer

@dataclass
class EvaluationMetrics:
    """Metrics for RAG evaluation"""
    total_questions: int = 0
    answerable_questions: int = 0
    unanswerable_questions: int = 0
    
    # For answerable questions
    correctly_answered: int = 0
    incorrectly_abstained: int = 0
    
    # For unanswerable questions  
    correctly_abstained: int = 0
    incorrectly_answered: int = 0
    
    # Quality metrics
    answer_relevance_scores: List[float] = None
    avg_confidence: float = 0.0
    
    def __post_init__(self):
        if self.answer_relevance_scores is None:
            self.answer_relevance_scores = []
    
    def calculate_summary(self) -> Dict[str, float]:
        """Calculate summary metrics"""
        answerable_accuracy = (
            self.correctly_answered / self.answerable_questions 
            if self.answerable_questions > 0 else 0
        )
        
        unanswerable_accuracy = (
            self.correctly_abstained / self.unanswerable_questions
            if self.unanswerable_questions > 0 else 0
        )
        
        overall_accuracy = (
            (self.correctly_answered + self.correctly_abstained) / self.total_questions
            if self.total_questions > 0 else 0
        )
        
        avg_relevance = (
            np.mean(self.answer_relevance_scores) 
            if self.answer_relevance_scores else 0
        )
        
        return {
            'overall_accuracy': overall_accuracy,
            'answerable_accuracy': answerable_accuracy,
            'unanswerable_accuracy': unanswerable_accuracy,
            'avg_relevance_score': avg_relevance,
            'avg_confidence': self.avg_confidence,
            'false_positive_rate': self.incorrectly_answered / max(self.unanswerable_questions, 1),
            'false_negative_rate': self.incorrectly_abstained / max(self.answerable_questions, 1)
        }

class GoldStandardEvaluator:
    """Evaluate the qa.py pipeline against the gold standard Q&A dataset."""

    def __init__(self, gold_data_path: str):
        self.gold_data = self._load_gold_data(gold_data_path)
        self.metrics = EvaluationMetrics()
        self.detailed_results = []
    
    def _load_gold_data(self, path: str) -> List[Dict]:
        """Load gold standard data from JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def evaluate(self, sample_size: int = None) -> Dict[str, Any]:
        """Run evaluation on gold standard dataset"""
        print("\n" + "="*60)
        print("EVALUATING RAG SYSTEM ON GOLD STANDARD DATASET")
        print("="*60)
        
        # Sample if requested
        eval_data = self.gold_data[:sample_size] if sample_size else self.gold_data
        
        # Separate by answerability
        answerable = [q for q in eval_data if q.get('answerable_gold', 0) == 1]
        unanswerable = [q for q in eval_data if q.get('answerable_gold', 0) == 0]
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  • Total questions: {len(eval_data)}")
        print(f"  • Answerable: {len(answerable)}")
        print(f"  • Unanswerable: {len(unanswerable)}")
        
        # Evaluate answerable questions
        print(f"\n🔍 Evaluating Answerable Questions...")
        self._evaluate_answerable(answerable)
        
        # Evaluate unanswerable questions
        print(f"\n🔍 Evaluating Unanswerable Questions...")
        self._evaluate_unanswerable(unanswerable)
        
        # Calculate final metrics
        self.metrics.total_questions = len(eval_data)
        self.metrics.answerable_questions = len(answerable)
        self.metrics.unanswerable_questions = len(unanswerable)
        
        summary = self.metrics.calculate_summary()
        
        # Display results
        self._display_results(summary)
        
        return {
            'metrics': summary,
            'detailed_results': self.detailed_results,
            'by_category': self._analyze_by_category()
        }
    
    @staticmethod
    def _is_abstained(result) -> bool:
        return result.answer.strip().lower().startswith("i don't know")

    def _evaluate_answerable(self, questions: List[Dict]):
        confidences = []

        for q_data in questions:
            question = q_data['question']
            gold_answer = q_data.get('gold_answer', '')
            category = q_data.get('category', 'unknown')

            response = qa_answer(question)
            abstained = self._is_abstained(response)

            if abstained:
                self.metrics.incorrectly_abstained += 1
                result_type = 'INCORRECT_ABSTENTION'
            else:
                relevance = self._calculate_answer_relevance(response.answer, gold_answer, question)
                if relevance > 0.3:
                    self.metrics.correctly_answered += 1
                    result_type = 'CORRECT'
                else:
                    self.metrics.incorrectly_abstained += 1
                    result_type = 'POOR_ANSWER'
                self.metrics.answer_relevance_scores.append(relevance)

            confidences.append(response.confidence)

            self.detailed_results.append({
                'question_id': q_data.get('id', 'unknown'),
                'question': question,
                'category': category,
                'answerable': True,
                'result_type': result_type,
                'confidence': response.confidence,
                'answer': response.answer,
                'gold_answer': gold_answer
            })

            if len(self.detailed_results) % 10 == 0:
                print(f"  Processed {len(self.detailed_results)} questions...")

        self.metrics.avg_confidence = np.mean(confidences) if confidences else 0

    def _evaluate_unanswerable(self, questions: List[Dict]):
        for q_data in questions:
            question = q_data['question']
            category = q_data.get('category', 'unknown')
            safety_tag = q_data.get('safety_tag', '')

            response = qa_answer(question)
            abstained = self._is_abstained(response)

            if abstained:
                self.metrics.correctly_abstained += 1
                result_type = 'CORRECT_ABSTENTION'
            else:
                self.metrics.incorrectly_answered += 1
                result_type = 'INCORRECT_ANSWER'

            self.detailed_results.append({
                'question_id': q_data.get('id', 'unknown'),
                'question': question,
                'category': category,
                'answerable': False,
                'result_type': result_type,
                'confidence': response.confidence,
                'answer': response.answer,
                'safety_tag': safety_tag
            })
    
    def _calculate_answer_relevance(self, generated: str, gold: str, question: str) -> float:
        """Calculate relevance score between generated and gold answers"""
        if not generated or not gold:
            return 0.0
        
        # Normalize texts
        gen_lower = generated.lower()
        gold_lower = gold.lower()
        
        # Extract key information
        key_terms = self._extract_key_terms(gold_lower)
        
        # Check for key term coverage
        coverage = sum(1 for term in key_terms if term in gen_lower) / max(len(key_terms), 1)
        
        # Check for numeric values
        gold_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:\s*%)?', gold))
        gen_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:\s*%)?', generated))
        number_match = len(gold_numbers & gen_numbers) / max(len(gold_numbers), 1) if gold_numbers else 1.0
        
        # Combine scores
        relevance = (coverage * 0.6 + number_match * 0.4)
        return relevance
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common words
        stopwords = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.split()
        key_terms = [w for w in words if len(w) > 3 and w not in stopwords]
        return key_terms[:10]  # Limit to top 10 terms
    
    def _analyze_by_category(self) -> Dict[str, Dict[str, float]]:
        """Analyze results by question category"""
        category_results = defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy': 0.0
        })
        
        for result in self.detailed_results:
            category = result['category']
            category_results[category]['total'] += 1
            
            if result['result_type'] in ['CORRECT', 'CORRECT_ABSTENTION']:
                category_results[category]['correct'] += 1
        
        # Calculate accuracy per category
        for cat in category_results:
            total = category_results[cat]['total']
            correct = category_results[cat]['correct']
            category_results[cat]['accuracy'] = correct / total if total > 0 else 0
        
        return dict(category_results)
    
    def _display_results(self, summary: Dict[str, float]):
        """Display evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print("\n📈 Overall Performance:")
        print(f"  • Overall Accuracy: {summary['overall_accuracy']:.2%}")
        print(f"  • Answerable Questions: {summary['answerable_accuracy']:.2%}")
        print(f"  • Unanswerable Questions: {summary['unanswerable_accuracy']:.2%}")
        
        print("\n📊 Quality Metrics:")
        print(f"  • Average Relevance Score: {summary['avg_relevance_score']:.2f}")
        print(f"  • Average Confidence: {summary['avg_confidence']:.2f}")
        
        print("\n⚠️ Error Analysis:")
        print(f"  • False Positive Rate: {summary['false_positive_rate']:.2%}")
        print(f"  • False Negative Rate: {summary['false_negative_rate']:.2%}")
    
    def save_results(self, output_path: str = "results/evaluation_results.json"):
        """Save detailed evaluation results"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'metrics': self.metrics.calculate_summary(),
            'by_category': self._analyze_by_category(),
            'detailed_results': self.detailed_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")

# ============= Training Data Generator =============
class TrainingDataGenerator:
    """Generate training data from gold standard for retriever fine-tuning"""
    
    def __init__(self, gold_data_path: str):
        self.gold_data = self._load_gold_data(gold_data_path)
    
    def _load_gold_data(self, path: str) -> List[Dict]:
        """Load gold standard data"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_training_pairs(self, chunks_path: str = "index/chunks.jsonl") -> str:
        """Generate training pairs for retriever fine-tuning"""
        
        # Load chunks
        chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        training_pairs = []
        
        # Process answerable questions
        for q_data in self.gold_data:
            if q_data.get('answerable_gold', 0) != 1:
                continue
            
            question = q_data['question']
            gold_answer = q_data.get('gold_answer', '')
            
            # Find positive chunks (those containing key information from gold answer)
            positive_chunks = self._find_positive_chunks(gold_answer, chunks)
            
            # Find hard negative chunks (similar but not containing the answer)
            negative_chunks = self._find_negative_chunks(question, chunks, positive_chunks)
            
            # Create training pairs
            for pos in positive_chunks[:2]:  # Use top 2 positive
                for neg in negative_chunks[:3]:  # Use top 3 negative
                    training_pairs.append({
                        'query': question,
                        'positive': pos['text'],
                        'negative': neg['text']
                    })
        
        # Save training data
        output_path = "train_retriever.tsv"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("query\tpositive\tnegative\n")
            for pair in training_pairs:
                f.write(f"{pair['query']}\t{pair['positive']}\t{pair['negative']}\n")
        
        print(f"✅ Generated {len(training_pairs)} training pairs")
        print(f"💾 Saved to {output_path}")
        
        return output_path
    
    def _find_positive_chunks(self, gold_answer: str, chunks: List[Dict]) -> List[Dict]:
        """Find chunks that contain the gold answer information"""
        gold_terms = set(re.findall(r'\b\w+\b', gold_answer.lower()))
        
        scored_chunks = []
        for chunk in chunks:
            chunk_terms = set(re.findall(r'\b\w+\b', chunk['text'].lower()))
            overlap = len(gold_terms & chunk_terms) / max(len(gold_terms), 1)
            
            if overlap > 0.3:  # Threshold for positive
                scored_chunks.append((overlap, chunk))
        
        # Sort by overlap score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:5]]
    
    def _find_negative_chunks(self, question: str, chunks: List[Dict], 
                              positive_chunks: List[Dict]) -> List[Dict]:
        """Find hard negative chunks"""
        question_terms = set(re.findall(r'\b\w+\b', question.lower()))
        positive_ids = {c.get('id', '') for c in positive_chunks}
        
        negative_chunks = []
        for chunk in chunks:
            # Skip if it's a positive chunk
            if chunk.get('id', '') in positive_ids:
                continue
            
            # Check if it has some relevance (hard negative)
            chunk_terms = set(re.findall(r'\b\w+\b', chunk['text'].lower()))
            overlap = len(question_terms & chunk_terms)
            
            if 1 <= overlap <= 3:  # Some overlap but not too much
                negative_chunks.append(chunk)
        
        return negative_chunks[:10]

# ============= Main Evaluation Script =============
def main():
    print("""
╔════════════════════════════════════════════════════════╗
║     QA PIPELINE EVALUATION WITH GOLD STANDARD DATASET  ║
╚════════════════════════════════════════════════════════╝
    """)

    gold_path = BASE_DIR / "data/eval/gold_diabetes_80.json"
    if not gold_path.exists():
        print("❌ Gold standard data not found!")
        print(f"  Expected: {gold_path}")
        return

    evaluator = GoldStandardEvaluator(str(gold_path))
    
    sample_size = None
    numeric_args = [a for a in sys.argv[1:] if a.isdigit()]
    if numeric_args:
        sample_size = int(numeric_args[0])
        print(f"\n📊 Running evaluation on {sample_size} samples...")
    else:
        print("\n📊 Running evaluation on full dataset...")
    
    results = evaluator.evaluate(sample_size)
    
    # Save results
    evaluator.save_results()
    
    # Analyze by category
    print("\n📂 Performance by Category:")
    for category, metrics in results['by_category'].items():
        print(f"  • {category:20s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    
    # Generate training data if requested
    if '--generate-training' in sys.argv:
        print("\n🏋️ Generating training data for retriever fine-tuning...")
        generator = TrainingDataGenerator(gold_path)
        generator.generate_training_pairs()
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()