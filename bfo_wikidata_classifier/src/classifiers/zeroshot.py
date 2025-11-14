"""
Zero-shot classifier using NLI models

Uses natural language inference to classify entities without training data.
"""

from typing import List, Optional
from transformers import pipeline

from classifiers.base import BaseClassifier
from models.wikidata import WikidataEntity
from models.results import ClassificationMatch


class ZeroShotClassifier(BaseClassifier):
    """Zero-shot classifier using transformers NLI"""

    def __init__(self, ontology, config: dict = None, model=None):
        """
        Initialize zero-shot classifier

        Args:
            ontology: BFO ontology
            config: Configuration dict
            model: Pre-loaded pipeline (or None to load default)
        """
        super().__init__(ontology, config)

        # Load or use provided model
        if model is None:
            model_name = self.config.get('model_name', 'facebook/bart-large-mnli')
            print(f"Loading zero-shot model: {model_name}")
            self.model = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1  # CPU
            )
        else:
            self.model = model

        # Get hypothesis template
        # Note: transformers expects {} not {class_label}
        self.hypothesis_template = self.config.get(
            'hypothesis_template',
            "This entity is {}."
        )

        # Prepare candidate labels
        self.candidate_labels = []
        self.label_to_uri = {}
        self._prepare_candidates()

    def _prepare_candidates(self):
        """Prepare candidate labels for zero-shot classification"""
        classes = self.ontology.get_all_classes()

        for cls in classes:
            # Use label as candidate
            label = cls.label

            # Also create natural language version of definition
            # E.g., "a material entity" instead of just "MaterialEntity"
            natural_label = self._make_natural_label(cls.label, cls.definition)

            self.candidate_labels.append(natural_label)
            self.label_to_uri[natural_label] = cls.uri

    def _make_natural_label(self, label: str, definition: str) -> str:
        """
        Create natural language label for hypothesis

        Args:
            label: Class label (e.g., "MaterialEntity")
            definition: Class definition

        Returns:
            Natural language label (e.g., "a material entity")
        """
        # Simple heuristic: use first sentence of definition
        first_sentence = definition.split('.')[0].strip()

        # If definition starts with "A/An", use it
        if first_sentence.lower().startswith(('a ', 'an ')):
            return first_sentence.lower()

        # Otherwise, prepend article to label
        label_lower = label.lower()

        # Add spaces before capitals (CamelCase -> camel case)
        import re
        spaced_label = re.sub(r'([A-Z])', r' \1', label).strip().lower()

        # Add article
        if spaced_label[0] in 'aeiou':
            return f"an {spaced_label}"
        else:
            return f"a {spaced_label}"

    def classify(self, entity: WikidataEntity, top_k: int = 3) -> List[ClassificationMatch]:
        """
        Classify using zero-shot NLI

        Args:
            entity: Wikidata entity
            top_k: Number of results

        Returns:
            List of matches
        """
        entity_text = entity.get_text()

        # Prepare hypothesis template with entity text if template uses {entity_text}
        if '{entity_text}' in self.hypothesis_template:
            # Template like: "{entity_text} is {}."
            # First replace {entity_text}, then the model will replace {} with class labels
            hypothesis_with_entity = self.hypothesis_template.replace('{entity_text}', entity_text)
        else:
            # Fallback to old template format: "This entity is {}."
            hypothesis_with_entity = self.hypothesis_template

        # Run zero-shot classification
        result = self.model(
            entity_text,
            self.candidate_labels,
            multi_label=self.config.get('multi_label', True),
            hypothesis_template=hypothesis_with_entity
        )

        # Parse results
        matches = []
        for label, score in zip(result['labels'][:top_k], result['scores'][:top_k]):
            uri = self.label_to_uri.get(label)
            if uri:
                bfo_class = self.ontology.get_class(uri)
                if bfo_class:
                    # For metadata, show the full hypothesis that was actually used
                    full_hypothesis = hypothesis_with_entity.format(label)

                    matches.append(ClassificationMatch(
                        class_uri=bfo_class.uri,
                        class_label=bfo_class.label,
                        confidence=float(score),
                        source=self.get_source_name(),
                        metadata={
                            'nli_score': float(score),
                            'hypothesis': full_hypothesis
                        }
                    ))

        return matches
