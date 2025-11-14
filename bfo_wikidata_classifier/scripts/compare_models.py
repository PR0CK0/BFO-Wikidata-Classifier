"""
Demo script: Compare different models on the same entity

Usage:
    python scripts/compare_models.py Q7186
    python scripts/compare_models.py "Marie Curie" --sample
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ontology import BFOOntology
from models.wikidata import WikidataAPI, create_sample_entities
from classifiers.semantic import SemanticClassifier
from classifiers.zeroshot import ZeroShotClassifier
from utils.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="Compare different models")
    parser.add_argument('entity', help="Wikidata ID or entity name if --sample")
    parser.add_argument('--sample', action='store_true',
                       help="Use sample entities")
    parser.add_argument('--type', default='semantic',
                       choices=['semantic', 'zeroshot'],
                       help="Model type to compare")
    parser.add_argument('--top-k', type=int, default=3,
                       help="Number of results")
    parser.add_argument('--cpu-only', action='store_true',
                       help="Force CPU-only inference (disable GPU)")

    args = parser.parse_args()

    print("=" * 80)
    print("MODEL COMPARISON")
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

    # Get entity
    print(f"\n2. Fetching entity...")
    if args.sample:
        sample_entities = {e.label: e for e in create_sample_entities()}
        entity = sample_entities.get(args.entity)
        if not entity:
            print(f"   Entity not found. Available: {', '.join(sample_entities.keys())}")
            return
    else:
        api = WikidataAPI()
        entity = api.fetch_entity(args.entity)

    print(f"   Entity: {entity.label}")
    print(f"   Description: {entity.description}")

    # Compare models
    print(f"\n3. Comparing {args.type} models...")
    registry = ModelRegistry(device=device)
    models = registry.list_available_models(args.type)

    print("\n" + "=" * 80)
    print(f"{args.type.upper()} MODEL COMPARISON")
    print("=" * 80)

    results = {}

    for model_info in models:
        model_name = model_info['name']
        print(f"\nTesting {model_name}...")
        print(f"  Size: {model_info['size']}, Speed: {model_info['speed']}, Quality: {model_info['quality']}")

        try:
            # Load model
            if args.type == 'semantic':
                model = registry.get_semantic_model(model_name)
                classifier = SemanticClassifier(ontology, {}, model=model)
            elif args.type == 'zeroshot':
                model = registry.get_zeroshot_model(model_name)
                classifier = ZeroShotClassifier(ontology, {}, model=model)

            # Classify
            start = time.time()
            matches = classifier.classify(entity, top_k=args.top_k)
            elapsed = (time.time() - start) * 1000

            results[model_name] = {
                'matches': matches,
                'time_ms': elapsed,
                'info': model_info
            }

            print(f"  Time: {elapsed:.1f}ms")
            print(f"  Top predictions:")
            for i, match in enumerate(matches, 1):
                print(f"    {i}. {match.class_label} ({match.confidence:.3f})")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<50} {'Time (ms)':<12} {'Top Prediction':<20} {'Conf.'}")
    print("-" * 80)

    for model_name, result in results.items():
        time_ms = result['time_ms']
        top_match = result['matches'][0] if result['matches'] else None

        if top_match:
            top_pred = top_match.class_label[:18]
            confidence = top_match.confidence
            print(f"{model_name:<50} {time_ms:>8.1f}      {top_pred:<20} {confidence:.3f}")
        else:
            print(f"{model_name:<50} {time_ms:>8.1f}      (no predictions)")

    print("=" * 80)

    # Speed vs Quality visualization
    print("\nSpeed Comparison (lower is better):")
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['time_ms']):
        time_ms = result['time_ms']
        bar_length = int(min(time_ms / 10, 50))
        bar = '█' * bar_length
        print(f"  {model_name[:30]:<30} {bar} {time_ms:.0f}ms")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
