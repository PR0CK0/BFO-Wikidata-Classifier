"""
Demo script: Evaluate all classifiers and strategies

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --generate-data
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ontology import BFOOntology
from classifiers.hybrid import HybridClassifier
from utils.model_registry import ModelRegistry
from utils.synthetic_data import generate_synthetic_examples, save_synthetic_data, load_synthetic_data
from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifiers and strategies")
    parser.add_argument('--generate-data', action='store_true',
                       help="Generate new synthetic data")
    parser.add_argument('--data-path', default='data/synthetic_labels.json',
                       help="Path to synthetic data")
    parser.add_argument('--config', help="Path to config file")
    parser.add_argument('--top-k', type=int, default=3,
                       help="Number of predictions")
    parser.add_argument('--preset', default='production',
                       choices=['production', 'research', 'resource_constrained', 'multilingual', 'ultra_lightweight'],
                       help="Model preset to use (default: production)")
    parser.add_argument('--cpu-only', action='store_true',
                       help="Force CPU-only inference (disable GPU)")

    args = parser.parse_args()

    print("=" * 80)
    print("BFO-WIKIDATA CLASSIFIER - EVALUATION")
    print("=" * 80)

    # Set device
    import torch
    if args.cpu_only:
        device = torch.device('cpu')
        print("   Device: CPU-only mode")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")

    # Load ontology
    print("\n1. Loading BFO ontology...")
    ontology_path = Path(__file__).parent.parent / "ontologies" / "bfo-2020.ttl"
    ontology = BFOOntology(str(ontology_path))
    print(f"   Loaded {len(ontology.get_all_classes())} classes from {ontology_path.name}")

    # Load or generate test data
    data_path = Path(args.data_path)

    if args.generate_data or not data_path.exists():
        print(f"\n2. Generating synthetic test data...")
        test_data = generate_synthetic_examples(ontology)

        # Save to file
        data_path.parent.mkdir(parents=True, exist_ok=True)
        save_synthetic_data(test_data, str(data_path))
    else:
        print(f"\n2. Loading test data from {data_path}...")
        test_data = load_synthetic_data(str(data_path))

    print(f"   Loaded {len(test_data)} test examples")

    # Load configuration
    print("\n3. Loading configuration...")
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "classification.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load preset configuration
    model_config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    preset = model_config['presets'][args.preset]
    config['models'] = {
        'semantic': preset['semantic'],
        'zeroshot': preset.get('zeroshot'),
        'finetuned_base': preset['finetuned_base']
    }

    print(f"   Using preset: {args.preset}")
    print(f"   Semantic model: {config['models']['semantic']}")
    print(f"   Zero-shot model: {config['models']['zeroshot'] or 'Disabled'}")

    # Create classifier
    print("\n4. Initializing classifiers...")
    registry = ModelRegistry(device=device)
    classifier = HybridClassifier(ontology, config, model_registry=registry)
    print(f"   Initialized {len(classifier.classifiers)} classifiers")

    # Create evaluator
    evaluator = Evaluator(ontology, classifier, test_data)

    # Evaluate individual classifiers
    print("\n5. Evaluating individual classifiers...")
    print("=" * 80)
    individual_results = evaluator.evaluate_individual_classifiers(top_k=args.top_k)
    evaluator.print_results(individual_results)

    # Evaluate hybrid strategies
    print("\n6. Evaluating hybrid strategies...")
    print("=" * 80)

    strategies = ['cascade', 'ensemble', 'hybrid_confidence', 'tiered']

    # Only test strategies for which we have the necessary classifiers
    if 'zeroshot' not in classifier.classifiers:
        print("   Note: Skipping ensemble and cascade (zero-shot disabled)")
        strategies = ['hybrid_confidence', 'tiered']

    strategy_results = evaluator.evaluate_all_strategies(
        strategies=strategies,
        top_k=args.top_k
    )
    evaluator.print_results(strategy_results)

    # Visualization
    if strategy_results:
        print(evaluator.compare_strategies_visual(strategy_results))

    # Save results
    print("\n7. Saving results...")
    results_path = Path("results") / "evaluation_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Individual Classifiers:\n")
        for name, metrics in individual_results.items():
            f.write(f"  {name}: {metrics['accuracy']*100:.2f}% ({metrics['avg_time_ms']:.1f}ms)\n")

        f.write("\nHybrid Strategies:\n")
        for name, metrics in strategy_results.items():
            f.write(f"  {name}: {metrics['accuracy']*100:.2f}% ({metrics['avg_time_ms']:.1f}ms)\n")

    print(f"   Results saved to {results_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
