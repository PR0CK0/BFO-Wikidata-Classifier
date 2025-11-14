"""
Hybrid classification strategies

Combines multiple classifiers using different strategies:
1. Cascade: Try classifiers in order, early exit on high confidence
2. Ensemble: Weighted average of all classifier scores
3. Hybrid-Confidence: Boost confidence when classifiers agree
4. Tiered: Adaptive strategy based on input characteristics
"""

import time
from typing import List, Dict, Optional
from collections import defaultdict

from classifiers.base import BaseClassifier
from classifiers.rule_based import RuleBasedClassifier
from classifiers.semantic import SemanticClassifier
from classifiers.zeroshot import ZeroShotClassifier
from classifiers.finetuned import FineTunedClassifier

from models.wikidata import WikidataEntity
from models.results import ClassificationMatch, ClassificationResult
from models.ontology import BFOOntology
from utils.model_registry import ModelRegistry


class HybridClassifier:
    """
    Hybrid classifier combining multiple classification approaches
    """

    def __init__(
        self,
        ontology: BFOOntology,
        config: dict,
        model_registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize hybrid classifier

        Args:
            ontology: BFO ontology
            config: Configuration dictionary
            model_registry: Model registry for loading specific models
        """
        self.ontology = ontology
        self.config = config
        self.registry = model_registry or ModelRegistry()

        # Initialize individual classifiers
        self.classifiers: Dict[str, BaseClassifier] = {}
        self._init_classifiers()

    def _init_classifiers(self):
        """Initialize individual classifiers based on config"""

        # Rule-based classifier
        if self.config.get('classifiers', {}).get('rule_based', {}).get('enabled', True):
            self.classifiers['rule_based'] = RuleBasedClassifier(
                self.ontology,
                self.config.get('classifiers', {}).get('rule_based', {})
            )

        # Semantic classifier
        if self.config.get('classifiers', {}).get('semantic', {}).get('enabled', True):
            semantic_model_name = self.config.get('models', {}).get('semantic')
            semantic_model = self.registry.get_semantic_model(semantic_model_name)

            self.classifiers['semantic'] = SemanticClassifier(
                self.ontology,
                self.config.get('classifiers', {}).get('semantic', {}),
                model=semantic_model
            )

        # Zero-shot classifier
        if self.config.get('classifiers', {}).get('zeroshot', {}).get('enabled', True):
            zeroshot_model_name = self.config.get('models', {}).get('zeroshot')
            if zeroshot_model_name:  # Only load if specified (can be None for resource-constrained)
                zeroshot_model = self.registry.get_zeroshot_model(zeroshot_model_name)

                self.classifiers['zeroshot'] = ZeroShotClassifier(
                    self.ontology,
                    self.config.get('classifiers', {}).get('zeroshot', {}),
                    model=zeroshot_model
                )

        # Fine-tuned classifier (stub)
        if self.config.get('classifiers', {}).get('finetuned', {}).get('enabled', False):
            base_model = self.config.get('models', {}).get('finetuned_base')

            self.classifiers['finetuned'] = FineTunedClassifier(
                self.ontology,
                self.config.get('classifiers', {}).get('finetuned', {}),
                base_model=base_model
            )

    def classify(
        self,
        entity: WikidataEntity,
        strategy: str = 'cascade',
        top_k: int = 3,
        hierarchical: bool = False
    ) -> ClassificationResult:
        """
        Classify entity using specified strategy

        Args:
            entity: Wikidata entity
            strategy: One of 'cascade', 'ensemble', 'hybrid_confidence', 'tiered'
            top_k: Number of results
            hierarchical: If True, use hierarchical classification (top-down)

        Returns:
            ClassificationResult
        """
        start_time = time.time()

        if hierarchical:
            # Use hierarchical top-down classification
            matches = self._classify_hierarchical(entity, strategy, top_k)
        else:
            # Use flat classification across all classes
            if strategy == 'cascade':
                matches = self._classify_cascade(entity, top_k)
            elif strategy == 'ensemble':
                matches = self._classify_ensemble(entity, top_k)
            elif strategy == 'hybrid_confidence':
                matches = self._classify_hybrid_confidence(entity, top_k)
            elif strategy == 'tiered':
                matches = self._classify_tiered(entity, top_k)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Infer parent classes if enabled (only for flat classification)
        parent_matches = []
        if not hierarchical and self.config.get('infer_parent_classes', True):
            parent_matches = self._infer_parent_classes(matches)

        return ClassificationResult(
            entity_id=entity.id,
            entity_label=entity.label,
            matches=matches,
            parent_matches=parent_matches,
            strategy=f"{strategy}_hierarchical" if hierarchical else strategy,
            processing_time_ms=processing_time
        )

    def _classify_hierarchical(
        self,
        entity: WikidataEntity,
        strategy: str,
        top_k: int
    ) -> List[ClassificationMatch]:
        """
        Hierarchical top-down classification

        Starts at root (Entity), progressively narrows down to more specific classes.
        Stops when confidence drops below threshold or reaches leaf node.

        Args:
            entity: Wikidata entity
            strategy: Base strategy to use at each level
            top_k: Number of results to track

        Returns:
            List of classification matches with hierarchical path
        """
        # Get hierarchical config
        hier_config = self.config.get('hierarchical', {})
        min_confidence = hier_config.get('min_confidence', 0.50)
        base_drop_threshold = hier_config.get('confidence_drop_threshold', 0.15)

        # Start at the root: BFO:Entity
        root_uri = "http://purl.obolibrary.org/obo/BFO_0000001"
        root_class = self.ontology.get_class(root_uri)

        if not root_class:
            # Fallback to flat classification if root not found
            return self._classify_by_strategy(entity, strategy, top_k)

        # Start at Entity root
        path = []
        current_node = root_class
        current_confidence = 1.0  # Start with full confidence at Entity
        current_depth = 0

        # Add Entity to the path as starting point
        path.append({
            'class_uri': root_class.uri,
            'class_label': root_class.label,
            'confidence': current_confidence,
            'decision': 'START',
            'depth': current_depth
        })

        current_depth += 1

        while current_node:
            # Get direct children of current node
            children = self.ontology.get_children(current_node.uri)

            # If no children, this is a leaf node - stop here
            if not children:
                path.append({
                    'class_uri': current_node.uri,
                    'class_label': current_node.label,
                    'confidence': current_confidence,
                    'decision': 'LEAF_NODE'
                })
                break

            # Classify among children only - get ALL children results
            child_results = self._classify_among_candidates(
                entity,
                children,
                strategy,
                top_k=len(children)  # Get all children, not just top_k
            )

            if not child_results:
                # No good classification among children - stop at parent
                path.append({
                    'class_uri': current_node.uri,
                    'class_label': current_node.label,
                    'confidence': current_confidence,
                    'decision': 'NO_CHILD_MATCH'
                })
                break

            # Debug: print all children results
            print(f"\n  Children of {current_node.label}:")
            for i, child in enumerate(child_results, 1):
                print(f"    {i}. {child.class_label}: {child.confidence:.4f}")

            # Get best child match
            best_child = child_results[0]

            # Check if confidence is acceptable
            if best_child.confidence < min_confidence:
                # Confidence too low - stop at current node
                path.append({
                    'class_uri': current_node.uri,
                    'class_label': current_node.label,
                    'confidence': current_confidence,
                    'decision': 'LOW_CONFIDENCE',
                    'attempted_child': best_child.class_label,
                    'child_confidence': best_child.confidence
                })
                break

            # Adaptive confidence drop threshold based on depth
            # Allow bigger drops at shallow depths (near root), stricter at deeper levels
            # Depth 0-1: More permissive
            # Depth 2+: use configured threshold (default 0.15)
            if current_depth <= 1:
                adaptive_threshold = 0.50  # Very permissive for first levels
            else:
                adaptive_threshold = base_drop_threshold

            confidence_drop = current_confidence - best_child.confidence

            # Check for confidence drop (child is much less confident than parent)
            if confidence_drop > adaptive_threshold:
                # Big confidence drop - stop at parent
                path.append({
                    'class_uri': current_node.uri,
                    'class_label': current_node.label,
                    'confidence': current_confidence,
                    'decision': 'CONFIDENCE_DROP',
                    'attempted_child': best_child.class_label,
                    'child_confidence': best_child.confidence,
                    'drop': confidence_drop,
                    'threshold': adaptive_threshold
                })
                break

            # Accept this child and continue down
            path.append({
                'class_uri': best_child.class_uri,
                'class_label': best_child.class_label,
                'confidence': best_child.confidence,
                'decision': 'CONTINUE',
                'depth': current_depth
            })

            # Move to this child for next iteration
            current_node = self.ontology.get_class(best_child.class_uri)
            current_confidence = best_child.confidence
            current_depth += 1

        # Build final results from the path
        if not path:
            # Shouldn't happen, but fallback
            return self._classify_by_strategy(entity, strategy, top_k)

        # The final node in the path is our classification
        final_node = path[-1]

        # Create the main match
        main_match = ClassificationMatch(
            class_uri=final_node['class_uri'],
            class_label=final_node['class_label'],
            confidence=final_node['confidence'],
            source=f'hierarchical_{strategy}',
            metadata={
                'hierarchical_path': path,
                'stop_reason': final_node['decision'],
                'depth': len(path)
            }
        )

        # Return top_k results (for now just the main one, could add siblings)
        return [main_match]

    def _classify_among_candidates(
        self,
        entity: WikidataEntity,
        candidate_uris: List[str],
        strategy: str,
        top_k: int
    ) -> List[ClassificationMatch]:
        """
        Classify entity among a specific set of candidate classes

        Uses semantic similarity to rank the candidate classes.

        Args:
            entity: Wikidata entity
            candidate_uris: List of BFO class URIs to consider
            strategy: Strategy to use (ignored for now, always uses semantic)
            top_k: Number of results

        Returns:
            Classification matches filtered to candidate set
        """
        # Use semantic classifier for hierarchical classification
        if 'semantic' not in self.classifiers:
            return []

        classifier = self.classifiers['semantic']

        # Get full classification, then filter to candidates
        all_matches = classifier.classify(entity, top_k=50)

        # Filter to only candidate URIs
        candidate_set = set(candidate_uris)
        filtered_matches = [m for m in all_matches if m.class_uri in candidate_set]

        # Update source to show hierarchical
        for match in filtered_matches:
            match.source = 'hierarchical_semantic'

        return filtered_matches[:top_k]

    def _classify_by_strategy(
        self,
        entity: WikidataEntity,
        strategy: str,
        top_k: int
    ) -> List[ClassificationMatch]:
        """Helper to call the appropriate strategy method"""
        if strategy == 'cascade':
            return self._classify_cascade(entity, top_k)
        elif strategy == 'ensemble':
            return self._classify_ensemble(entity, top_k)
        elif strategy == 'hybrid_confidence':
            return self._classify_hybrid_confidence(entity, top_k)
        elif strategy == 'tiered':
            return self._classify_tiered(entity, top_k)
        else:
            # Default to cascade
            return self._classify_cascade(entity, top_k)

    def _classify_cascade(self, entity: WikidataEntity, top_k: int) -> List[ClassificationMatch]:
        """
        Cascade strategy: Try classifiers in order, exit early on high confidence

        Order: rule_based -> semantic -> zeroshot -> finetuned
        """
        strategy_config = self.config.get('strategies', {}).get('cascade', {})
        thresholds = strategy_config.get('confidence_thresholds', {
            'rule_based': 0.90,
            'semantic': 0.80,
            'zeroshot': 0.70
        })

        order = strategy_config.get('order', ['rule_based', 'semantic', 'zeroshot', 'finetuned'])

        # Track cascade decisions for logging
        cascade_log = []
        last_matches = []

        for classifier_name in order:
            if classifier_name not in self.classifiers:
                continue

            classifier = self.classifiers[classifier_name]
            matches = classifier.classify(entity, top_k=top_k)

            if matches:
                last_matches = matches
                # Check if top match exceeds threshold
                top_confidence = matches[0].confidence
                threshold = thresholds.get(classifier_name, 0.7)

                # Log this decision
                decision = {
                    'classifier': classifier_name,
                    'top_prediction': matches[0].class_label,
                    'confidence': float(top_confidence),
                    'threshold': threshold,
                    'decision': 'ACCEPT' if top_confidence >= threshold else 'CONTINUE'
                }
                cascade_log.append(decision)

                if top_confidence >= threshold:
                    # Early exit - add cascade log to metadata
                    if matches:
                        matches[0].metadata['cascade_decisions'] = cascade_log
                    return matches
            else:
                cascade_log.append({
                    'classifier': classifier_name,
                    'top_prediction': None,
                    'confidence': 0.0,
                    'threshold': thresholds.get(classifier_name, 0.7),
                    'decision': 'NO_RESULTS'
                })

        # If no early exit, return last non-empty result
        if last_matches:
            last_matches[0].metadata['cascade_decisions'] = cascade_log
            return last_matches

        # Try semantic as fallback (most reliable)
        if 'semantic' in self.classifiers:
            return self.classifiers['semantic'].classify(entity, top_k=top_k)

        return []

    def _classify_ensemble(self, entity: WikidataEntity, top_k: int) -> List[ClassificationMatch]:
        """
        Ensemble strategy: Weighted average of all classifiers
        """
        strategy_config = self.config.get('strategies', {}).get('ensemble', {})
        weights = strategy_config.get('weights', {
            'rule_based': 0.15,
            'semantic': 0.50,
            'zeroshot': 0.35,
            'finetuned': 0.0
        })

        # Collect results from all classifiers
        all_results: Dict[str, List[ClassificationMatch]] = {}

        for classifier_name, classifier in self.classifiers.items():
            if weights.get(classifier_name, 0) > 0:
                all_results[classifier_name] = classifier.classify(entity, top_k=top_k * 2)

        # Aggregate scores by class URI
        class_scores: Dict[str, float] = defaultdict(float)
        class_labels: Dict[str, str] = {}
        class_sources: Dict[str, List[str]] = defaultdict(list)

        for classifier_name, matches in all_results.items():
            weight = weights.get(classifier_name, 0)
            for match in matches:
                class_scores[match.class_uri] += match.confidence * weight
                class_labels[match.class_uri] = match.class_label
                class_sources[match.class_uri].append(f"{classifier_name}:{match.confidence:.2f}")

        # Sort by aggregated score
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for uri, score in sorted_classes[:top_k]:
            results.append(ClassificationMatch(
                class_uri=uri,
                class_label=class_labels[uri],
                confidence=score,
                source='ensemble',
                metadata={'sources': class_sources[uri]}
            ))

        return results

    def _classify_hybrid_confidence(self, entity: WikidataEntity, top_k: int) -> List[ClassificationMatch]:
        """
        Hybrid-confidence strategy: Boost confidence when classifiers agree
        """
        strategy_config = self.config.get('strategies', {}).get('hybrid_confidence', {})
        base_classifiers = strategy_config.get('base_classifiers', ['rule_based', 'semantic'])
        agreement_boost = strategy_config.get('agreement_boost', 0.15)

        # Get results from base classifiers
        all_results = {}
        for classifier_name in base_classifiers:
            if classifier_name in self.classifiers:
                all_results[classifier_name] = self.classifiers[classifier_name].classify(
                    entity, top_k=top_k * 2
                )

        # Find agreeing predictions
        class_votes: Dict[str, List[float]] = defaultdict(list)
        class_labels: Dict[str, str] = {}

        for classifier_name, matches in all_results.items():
            for match in matches:
                class_votes[match.class_uri].append(match.confidence)
                class_labels[match.class_uri] = match.class_label

        # Calculate boosted scores
        results = []
        for uri, confidences in class_votes.items():
            # Base score: max confidence from any classifier
            base_score = max(confidences)

            # Boost if multiple classifiers agree
            if len(confidences) >= strategy_config.get('min_agreement', 2):
                boosted_score = min(base_score + agreement_boost, 1.0)
            else:
                boosted_score = base_score

            results.append(ClassificationMatch(
                class_uri=uri,
                class_label=class_labels[uri],
                confidence=boosted_score,
                source='hybrid_confidence',
                metadata={
                    'agreement_count': len(confidences),
                    'base_score': base_score,
                    'boost_applied': len(confidences) >= 2
                }
            ))

        # Sort and return top-k
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def _classify_tiered(self, entity: WikidataEntity, top_k: int) -> List[ClassificationMatch]:
        """
        Tiered strategy: Adaptive based on input characteristics
        """
        # Determine appropriate strategy based on entity characteristics
        if entity.has_many_aliases():
            # Well-known entity -> use fast cascade
            return self._classify_cascade(entity, top_k)
        elif entity.has_short_description():
            # Limited context -> use ensemble for accuracy
            return self._classify_ensemble(entity, top_k)
        else:
            # Default -> use semantic (fast and reliable)
            if 'semantic' in self.classifiers:
                matches = self.classifiers['semantic'].classify(entity, top_k=top_k)
                # Update source to reflect tiered decision
                for match in matches:
                    match.source = 'tiered'
                return matches

        return []

    def _infer_parent_classes(self, matches: List[ClassificationMatch]) -> List[ClassificationMatch]:
        """
        Infer parent classes from classified matches

        Args:
            matches: Primary classification matches

        Returns:
            List of parent class matches
        """
        parent_matches = []
        seen_uris = set(m.class_uri for m in matches)

        for match in matches:
            # Get all ancestors
            ancestors = self.ontology.get_ancestors(match.class_uri)

            # Skip self
            for ancestor_uri in ancestors:
                if ancestor_uri == match.class_uri or ancestor_uri in seen_uris:
                    continue

                ancestor_class = self.ontology.get_class(ancestor_uri)
                if ancestor_class:
                    # Parent has slightly lower confidence
                    parent_confidence = match.confidence * 0.9

                    parent_matches.append(ClassificationMatch(
                        class_uri=ancestor_class.uri,
                        class_label=ancestor_class.label,
                        confidence=parent_confidence,
                        source='inferred',
                        metadata={'inferred_from': match.class_uri}
                    ))

                    seen_uris.add(ancestor_uri)

        return parent_matches

    def compare_strategies(
        self,
        entity: WikidataEntity,
        strategies: List[str] = None,
        top_k: int = 3
    ) -> Dict[str, ClassificationResult]:
        """
        Compare multiple strategies on the same entity

        Args:
            entity: Wikidata entity
            strategies: List of strategies to compare
            top_k: Number of results per strategy

        Returns:
            Dictionary mapping strategy name to result
        """
        if strategies is None:
            strategies = ['cascade', 'ensemble', 'hybrid_confidence', 'tiered']

        results = {}
        for strategy in strategies:
            results[strategy] = self.classify(entity, strategy=strategy, top_k=top_k)

        return results
