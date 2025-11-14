"""
Evaluation framework for comparing classifiers and strategies
"""

import time
from typing import List, Dict, Tuple
from collections import defaultdict

from models.wikidata import WikidataEntity
from models.ontology import BFOOntology
from classifiers.hybrid import HybridClassifier


class Evaluator:
    """Evaluates classification performance on labeled data"""

    def __init__(
        self,
        ontology: BFOOntology,
        classifier: HybridClassifier,
        test_data: List[Tuple[WikidataEntity, str]]
    ):
        """
        Initialize evaluator

        Args:
            ontology: BFO ontology
            classifier: Hybrid classifier to evaluate
            test_data: List of (entity, ground_truth_uri) tuples
        """
        self.ontology = ontology
        self.classifier = classifier
        self.test_data = test_data

    def evaluate_strategy(
        self,
        strategy: str,
        top_k: int = 3,
        include_parents: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single strategy

        Args:
            strategy: Strategy name
            top_k: Number of predictions to consider
            include_parents: Whether to consider parent classes as correct

        Returns:
            Dictionary with metrics
        """
        correct = 0
        total = len(self.test_data)
        total_time = 0.0

        # Track per-class metrics
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for entity, ground_truth_uri in self.test_data:
            # Classify
            result = self.classifier.classify(entity, strategy=strategy, top_k=top_k)
            total_time += result.processing_time_ms

            # Check if ground truth in predictions
            predicted_uris = [m.class_uri for m in result.matches]

            if include_parents:
                # Also consider parent classes
                predicted_uris.extend([m.class_uri for m in result.parent_matches])

            is_correct = ground_truth_uri in predicted_uris

            if is_correct:
                correct += 1

            # Track per-class
            ground_truth_class = self.ontology.get_class(ground_truth_uri)
            if ground_truth_class:
                class_label = ground_truth_class.label
                class_total[class_label] += 1
                if is_correct:
                    class_correct[class_label] += 1

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_time_ms = total_time / total if total > 0 else 0.0

        # Per-class accuracy
        per_class_accuracy = {}
        for class_label in class_total.keys():
            per_class_accuracy[class_label] = (
                class_correct[class_label] / class_total[class_label]
                if class_total[class_label] > 0 else 0.0
            )

        return {
            'strategy': strategy,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_time_ms': avg_time_ms,
            'per_class_accuracy': per_class_accuracy
        }

    def evaluate_all_strategies(
        self,
        strategies: List[str] = None,
        top_k: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all strategies

        Args:
            strategies: List of strategy names (None = all)
            top_k: Number of predictions

        Returns:
            Dictionary mapping strategy to metrics
        """
        if strategies is None:
            strategies = ['cascade', 'ensemble', 'hybrid_confidence', 'tiered']

        results = {}
        for strategy in strategies:
            print(f"Evaluating {strategy}...")
            results[strategy] = self.evaluate_strategy(strategy, top_k=top_k)

        return results

    def evaluate_individual_classifiers(self, top_k: int = 3) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual classifiers (not hybrid)

        Args:
            top_k: Number of predictions

        Returns:
            Dictionary mapping classifier name to metrics
        """
        results = {}

        for classifier_name, classifier in self.classifier.classifiers.items():
            print(f"Evaluating {classifier_name}...")

            correct = 0
            total = len(self.test_data)
            total_time = 0.0

            for entity, ground_truth_uri in self.test_data:
                start = time.time()
                matches = classifier.classify(entity, top_k=top_k)
                elapsed = (time.time() - start) * 1000

                total_time += elapsed

                predicted_uris = [m.class_uri for m in matches]
                if ground_truth_uri in predicted_uris:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0
            avg_time_ms = total_time / total if total > 0 else 0.0

            results[classifier_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'avg_time_ms': avg_time_ms
            }

        return results

    def print_results(self, results: Dict[str, Dict[str, float]]):
        """
        Print evaluation results in formatted table

        Args:
            results: Results from evaluate_all_strategies or evaluate_individual_classifiers
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"{'Strategy/Classifier':<25} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Time (ms)'}")
        print("-" * 80)

        for name, metrics in results.items():
            accuracy = metrics['accuracy'] * 100
            correct = metrics['correct']
            total = metrics['total']
            avg_time = metrics['avg_time_ms']

            print(f"{name:<25} {accuracy:>6.2f}%      {correct:>3}/{total:<3}          {avg_time:>6.1f}")

        print("=" * 80)

        # Print per-class accuracy if available
        for name, metrics in results.items():
            if 'per_class_accuracy' in metrics:
                print(f"\nPer-class accuracy for {name}:")
                for class_label, acc in metrics['per_class_accuracy'].items():
                    print(f"  {class_label:<20} {acc*100:>6.2f}%")

    def compare_strategies_visual(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Create a simple ASCII visualization comparing strategies

        Args:
            results: Results dictionary

        Returns:
            ASCII art visualization
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("STRATEGY COMPARISON")
        lines.append("=" * 80)

        # Accuracy comparison
        lines.append("\nAccuracy (%):")
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            accuracy = metrics['accuracy'] * 100
            bar_length = int(accuracy / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            lines.append(f"  {name:<20} {bar} {accuracy:.1f}%")

        # Speed comparison
        lines.append("\nSpeed (lower is better, ms):")
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['avg_time_ms']):
            avg_time = metrics['avg_time_ms']
            bar_length = int(min(avg_time / 5, 50))  # Scale
            bar = '█' * bar_length
            lines.append(f"  {name:<20} {bar} {avg_time:.1f}ms")

        lines.append("=" * 80)
        return "\n".join(lines)


if __name__ == "__main__":
    from ..models.ontology import BFOOntology
    from ..utils.synthetic_data import generate_synthetic_examples
    from ..classifiers.hybrid import HybridClassifier
    from pathlib import Path

    # Create test setup
    ontology_path = Path(__file__).parent.parent.parent / "ontologies" / "bfo-2020.ttl"
    ontology = BFOOntology(str(ontology_path))
    test_data = generate_synthetic_examples(ontology)

    # Create classifier with minimal config
    config = {
        'classifiers': {
            'rule_based': {'enabled': True},
            'semantic': {'enabled': True},
            'zeroshot': {'enabled': False}  # Skip for demo
        },
        'strategies': {
            'cascade': {},
            'ensemble': {},
            'hybrid_confidence': {}
        }
    }

    classifier = HybridClassifier(ontology, config)

    # Evaluate
    evaluator = Evaluator(ontology, classifier, test_data)

    print("Evaluating individual classifiers...")
    individual_results = evaluator.evaluate_individual_classifiers()
    evaluator.print_results(individual_results)

    print("\nEvaluating hybrid strategies...")
    strategy_results = evaluator.evaluate_all_strategies(['cascade', 'ensemble'])
    evaluator.print_results(strategy_results)
    print(evaluator.compare_strategies_visual(strategy_results))
