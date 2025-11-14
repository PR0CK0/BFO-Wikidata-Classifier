"""
Demo script: Classify a single Wikidata entity

Usage:
    python scripts/classify_entity.py Q7186
    python scripts/classify_entity.py Q362 --strategy ensemble
    python scripts/classify_entity.py "Marie Curie" --sample
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ontology import BFOOntology
from models.wikidata import WikidataAPI, create_sample_entities
from classifiers.hybrid import HybridClassifier
from utils.model_registry import ModelRegistry
from utils.classification_logger import ClassificationLogger
import yaml


def load_config(config_path: str = None) -> dict:
    """Load configuration"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "classification.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Add model selection (use production preset by default)
    model_config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Use production preset
    preset = model_config['presets']['production']
    config['models'] = {
        'semantic': preset['semantic'],
        'zeroshot': preset.get('zeroshot'),
        'finetuned_base': preset['finetuned_base']
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Classify a Wikidata entity to BFO classes")
    parser.add_argument('entity', nargs='?', help="Wikidata ID (e.g., Q7186) or entity name if --sample")
    parser.add_argument('--strategy', default='cascade',
                       choices=['cascade', 'ensemble', 'hybrid_confidence', 'tiered'],
                       help="Hybrid strategy to use (default: cascade)")
    parser.add_argument('--sample', action='store_true',
                       help="Use sample entities instead of API")
    parser.add_argument('--random', action='store_true',
                       help="Classify a random Wikidata entity")
    parser.add_argument('--top-k', type=int, default=3,
                       help="Number of results to return (default: 3)")
    parser.add_argument('--compare', action='store_true',
                       help="Compare all strategies")
    parser.add_argument('--config', help="Path to config file")
    parser.add_argument('--preset', default='production',
                       choices=['production', 'research', 'resource_constrained', 'multilingual', 'ultra_lightweight'],
                       help="Model preset to use (default: production)")
    parser.add_argument('--semantic-model', help="Override semantic model (e.g., all-MiniLM-L6-v2)")
    parser.add_argument('--zeroshot-model', help="Override zero-shot model (e.g., facebook/bart-large-mnli)")
    parser.add_argument('--finetuned-model', help="Override fine-tuned base model (e.g., Prajjwal1/bert-tiny)")
    parser.add_argument('--cpu-only', action='store_true',
                       help="Force CPU-only inference (disable GPU)")
    parser.add_argument('--log', action='store_true',
                       help="Save classification results to log file (logs/ directory)")
    parser.add_argument('--log-dir', default='logs',
                       help="Directory to save log files (default: logs)")
    parser.add_argument('--hierarchical', action='store_true',
                       help="Use hierarchical classification (top-down from Entity)")

    args = parser.parse_args()

    # Validate arguments
    if not args.random and not args.entity:
        parser.error("entity argument is required unless --random is specified")

    print("=" * 80)
    print("BFO-WIKIDATA CLASSIFIER - SINGLE ENTITY DEMO")
    print("=" * 80)

    # Set device
    import torch
    if args.cpu_only:
        device = torch.device('cpu')
        print("   Device: CPU-only mode")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config(args.config)

    # Load preset configuration if specified
    if args.preset != 'production':  # Production is already loaded by default
        model_config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        preset = model_config['presets'][args.preset]
        config['models'] = {
            'semantic': preset['semantic'],
            'zeroshot': preset.get('zeroshot'),
            'finetuned_base': preset['finetuned_base']
        }

    # Apply CLI model overrides
    if args.semantic_model:
        config['models']['semantic'] = args.semantic_model
        print(f"   Overriding semantic model: {args.semantic_model}")
    if args.zeroshot_model:
        config['models']['zeroshot'] = args.zeroshot_model
        print(f"   Overriding zero-shot model: {args.zeroshot_model}")
    if args.finetuned_model:
        config['models']['finetuned_base'] = args.finetuned_model
        print(f"   Overriding fine-tuned base model: {args.finetuned_model}")

    print(f"   Using strategy: {args.strategy}")
    print(f"   Using preset: {args.preset}")
    print(f"   Semantic model: {config['models']['semantic']}")
    print(f"   Zero-shot model: {config['models']['zeroshot'] or 'Disabled'}")
    print(f"   Fine-tuned base: {config['models']['finetuned_base']}")

    # Load ontology
    print("\n2. Loading BFO ontology...")
    ontology_path = Path(__file__).parent.parent / "ontologies" / "bfo-2020.ttl"
    ontology = BFOOntology(str(ontology_path))
    print(f"   Loaded {len(ontology.get_all_classes())} classes from {ontology_path.name}")

    # Get entity
    print(f"\n3. Fetching entity...")
    if args.random:
        # Fetch random entity from Wikidata
        api = WikidataAPI()
        try:
            print("   Fetching random Wikidata entity...")
            entity = api.fetch_random_entity()
            print(f"   Found random entity: {entity.id}")
        except Exception as e:
            print(f"   Error fetching random entity: {e}")
            return
    elif args.sample:
        # Use sample data
        sample_entities = {e.label: e for e in create_sample_entities()}
        if args.entity in sample_entities:
            entity = sample_entities[args.entity]
        else:
            print(f"   Sample entity '{args.entity}' not found")
            print(f"   Available: {', '.join(sample_entities.keys())}")
            return
    else:
        # Fetch from API
        api = WikidataAPI()
        try:
            entity = api.fetch_entity(args.entity)
        except Exception as e:
            print(f"   Error fetching entity: {e}")
            return

    print(f"   Entity: {entity.label} ({entity.id})")
    print(f"   Description: {entity.description}")
    if entity.aliases:
        try:
            print(f"   Aliases: {', '.join(entity.aliases[:5])}")
        except UnicodeEncodeError:
            # Fallback for non-ASCII aliases that can't be encoded to console
            print(f"   Aliases: {len(entity.aliases)} alias(es) available")

    # Create classifier
    print("\n4. Initializing classifier...")
    registry = ModelRegistry(device=device)
    classifier = HybridClassifier(ontology, config, model_registry=registry)
    print(f"   Loaded {len(classifier.classifiers)} classifiers")

    # Classify
    print(f"\n5. Classifying...")

    if args.compare:
        # Compare all strategies
        print("   Comparing all strategies...")
        results = classifier.compare_strategies(entity, top_k=args.top_k)

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        for strategy_name, result in results.items():
            print(f"\n{strategy_name.upper()} ({result.processing_time_ms:.1f}ms):")
            for i, match in enumerate(result.matches, 1):
                print(f"  {i}. {match.class_label} ({match.confidence:.3f}) [{match.source}]")

    else:
        # Single strategy
        result = classifier.classify(
            entity,
            strategy=args.strategy,
            top_k=args.top_k,
            hierarchical=args.hierarchical
        )

        # Log if requested
        if args.log:
            logger = ClassificationLogger(log_dir=args.log_dir)
            log_path = logger.log_classification(
                entity=entity,
                result=result,
                config=config,
                device=str(device),
                additional_metadata={
                    'preset': args.preset,
                    'cli_overrides': {
                        'semantic_model': args.semantic_model,
                        'zeroshot_model': args.zeroshot_model,
                        'finetuned_model': args.finetuned_model
                    },
                    'sample_mode': args.sample,
                    'random_mode': args.random,
                    'hierarchical_mode': args.hierarchical
                }
            )
            print(f"\n[OK] Classification logged to: {log_path}")

        print("\n" + "=" * 80)
        print("CLASSIFICATION RESULTS")
        print("=" * 80)
        print(result.format_output())

        # Show hierarchical path if available
        if args.hierarchical and result.matches and 'hierarchical_path' in result.matches[0].metadata:
            print("\n" + "=" * 80)
            print("HIERARCHICAL CLASSIFICATION PATH")
            print("=" * 80)
            path = result.matches[0].metadata['hierarchical_path']
            depth = result.matches[0].metadata.get('depth', len(path))
            stop_reason = result.matches[0].metadata.get('stop_reason', 'UNKNOWN')

            print(f"Depth: {depth} levels")
            print(f"Stop Reason: {stop_reason}\n")

            for i, step in enumerate(path, 1):
                indent = "  " * (i - 1)
                decision_symbol = {
                    'START': '(root)',
                    'CONTINUE': '->',
                    'LEAF_NODE': '[LEAF]',
                    'LOW_CONFIDENCE': '[LOW_CONF]',
                    'CONFIDENCE_DROP': '[CONF_DROP]',
                    'NO_CHILD_MATCH': '[NO_CHILD]'
                }.get(step['decision'], step['decision'])

                print(f"{indent}Level {i}: {step['class_label']} (conf: {step['confidence']:.3f}) {decision_symbol}")

                # Show cascade trace if available
                if 'cascade_log' in step and step['cascade_log']:
                    cascade_indent = indent + "        "
                    print(f"{cascade_indent}Cascade:")
                    for cascade_step in step['cascade_log']:
                        decision_sym = {'ACCEPT': 'ACCEPT', 'CONTINUE': 'continue', 'NO_MATCH': 'no match', 'ERROR': 'error'}.get(cascade_step['decision'], cascade_step['decision'])
                        if cascade_step.get('top_prediction'):
                            print(f"{cascade_indent}  {cascade_step['classifier']}: {cascade_step['top_prediction']} ({cascade_step['confidence']:.3f}, threshold {cascade_step['threshold']:.2f}) -> {decision_sym}")
                        else:
                            print(f"{cascade_step['classifier']}: {decision_sym}")

                # Show additional details for stop decisions
                if 'attempted_child' in step:
                    print(f"{indent}        Tried: {step['attempted_child']} (conf: {step['child_confidence']:.3f})")
                if 'drop' in step:
                    threshold_str = f" (threshold: {step['threshold']:.2f})" if 'threshold' in step else ""
                    print(f"{indent}        Drop: {step['drop']:.3f}{threshold_str}")

        # Show cascade decision trace if available (only for non-hierarchical)
        elif result.matches and 'cascade_decisions' in result.matches[0].metadata:
            print("\n" + "=" * 80)
            print("CASCADE DECISION TRACE")
            print("=" * 80)
            cascade_log = result.matches[0].metadata['cascade_decisions']
            for step in cascade_log:
                decision_symbol = {
                    'ACCEPT': '[ACCEPTED]',
                    'CONTINUE': '-> Continue to next',
                    'NO_RESULTS': '[NO_RESULTS]'
                }.get(step['decision'], step['decision'])

                print(f"\n{step['classifier'].upper()}:")
                if step['top_prediction']:
                    print(f"  Prediction: {step['top_prediction']}")
                    print(f"  Confidence: {step['confidence']:.3f}")
                    print(f"  Threshold:  {step['threshold']:.3f}")
                    print(f"  Decision:   {decision_symbol}")
                else:
                    print(f"  Decision:   {decision_symbol}")

        # Show detailed match info
        print("\n" + "=" * 80)
        print("DETAILED MATCHES")
        print("=" * 80)
        for i, match in enumerate(result.matches, 1):
            bfo_class = ontology.get_class(match.class_uri)
            print(f"\n{i}. {match.class_label} (confidence: {match.confidence:.3f})")
            print(f"   URI: {match.class_uri}")
            print(f"   Source: {match.source}")
            if bfo_class:
                print(f"   Definition: {bfo_class.definition[:100]}...")

            # Show hypothesis for zero-shot matches
            if match.metadata and 'hypothesis' in match.metadata:
                print(f"   Hypothesis: {match.metadata['hypothesis']}")

            if match.metadata:
                # Don't print cascade_decisions or hypothesis again (already shown above)
                filtered_metadata = {k: v for k, v in match.metadata.items()
                                   if k not in ['cascade_decisions', 'hypothesis']}
                if filtered_metadata:
                    print(f"   Metadata: {filtered_metadata}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
