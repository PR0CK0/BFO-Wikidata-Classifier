"""
Rule-based classifier using keyword matching and Wikidata claims

Rules include:
1. Wikidata structural rules (P31 "instance of" property):
   - P31=Q5 (human) → MaterialEntity
   - More can be added...

2. Keyword matching rules:
   - Keywords like "process", "event" → bfo:Process
   - Keywords like "object", "entity" → bfo:MaterialEntity
   - Keywords like "quality", "property" → bfo:Quality
   - Keywords like "role", "function" → bfo:Role
"""

from typing import List, Dict, Optional
import re
from classifiers.base import BaseClassifier
from models.wikidata import WikidataEntity
from models.results import ClassificationMatch


class RuleBasedClassifier(BaseClassifier):
    """Rule-based classifier using keyword matching"""

    def __init__(self, ontology, config: dict = None):
        super().__init__(ontology, config)

        # Define Wikidata claim rules (P31 "instance of")
        # Maps Wikidata entity ID to BFO class label
        self.wikidata_rules: Dict[str, str] = {
            'Q5': 'MaterialEntity',  # human → MaterialEntity
            # Can add more mappings here:
            # 'Q16521': 'MaterialEntity',  # taxon → MaterialEntity
            # 'Q1656682': 'Process',  # event → Process
            # etc.
        }

        # Define keyword rules (class_label -> keywords)
        self.rules: Dict[str, List[str]] = {
            'Process': [
                'process', 'event', 'activity', 'action', 'happening',
                'war', 'revolution', 'movement', 'ceremony', 'competition',
                'development', 'growth', 'change', 'transformation'
            ],
            'MaterialEntity': [
                'object', 'entity', 'thing', 'material', 'substance',
                'person', 'people', 'organism', 'creature', 'being',
                'structure', 'building', 'device', 'machine', 'tool',
                'molecule', 'cell', 'particle', 'body'
            ],
            'Quality': [
                'quality', 'property', 'characteristic', 'attribute',
                'color', 'colour', 'shape', 'size', 'temperature',
                'mass', 'weight', 'density', 'speed', 'brightness'
            ],
            'Role': [
                'role', 'function', 'capacity', 'position', 'status',
                'occupation', 'profession', 'job', 'responsibility',
                'purpose', 'duty', 'task'
            ],
            'SpatialRegion': [
                'location', 'place', 'region', 'area', 'zone',
                'space', 'position', 'site', 'spot', 'territory',
                'country', 'city', 'continent'
            ],
            'IndependentContinuant': [
                'independent', 'standalone', 'autonomous', 'self-sufficient'
            ],
        }

        # Compile regex patterns for efficiency
        self.patterns: Dict[str, re.Pattern] = {}
        for class_label, keywords in self.rules.items():
            # Create pattern: \b(word1|word2|...)\b (word boundaries)
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.patterns[class_label] = re.compile(pattern, re.IGNORECASE)

    def _check_wikidata_claims(self, entity: WikidataEntity) -> List[ClassificationMatch]:
        """
        Check Wikidata claims for structural rules (high confidence)

        Args:
            entity: Wikidata entity

        Returns:
            List of matches based on Wikidata claims (empty if none found)
        """
        results = []

        # Check P31 (instance of) property
        if 'P31' in entity.claims:
            instance_of_values = entity.claims['P31']

            # Check each instance_of value against our rules
            for value_id in instance_of_values:
                if value_id in self.wikidata_rules:
                    class_label = self.wikidata_rules[value_id]

                    # Find corresponding BFO class
                    bfo_class = self.ontology.get_class_by_label(class_label)
                    if bfo_class:
                        results.append(ClassificationMatch(
                            class_uri=bfo_class.uri,
                            class_label=bfo_class.label,
                            confidence=0.95,  # High confidence for structural rules
                            source=self.get_source_name(),
                            metadata={
                                'rule_type': 'wikidata_claim',
                                'property': 'P31',
                                'value': value_id
                            }
                        ))

        return results

    def classify(self, entity: WikidataEntity, top_k: int = 3) -> List[ClassificationMatch]:
        """
        Classify using Wikidata claims and keyword rules

        Args:
            entity: Wikidata entity
            top_k: Number of results to return

        Returns:
            List of matches
        """
        # Check Wikidata claims first (high confidence structural rules)
        claim_matches = self._check_wikidata_claims(entity)
        if claim_matches:
            return claim_matches[:top_k]

        # Fall back to keyword matching
        text = entity.get_text().lower()

        # Score each class based on keyword matches
        scores: Dict[str, float] = {}

        for class_label, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Score = number of unique keywords matched
                unique_matches = set(matches)
                score = len(unique_matches) / len(self.rules[class_label])
                scores[class_label] = score

        # If no matches, return empty
        if not scores:
            return []

        # Convert to ClassificationMatch objects
        results = []
        for class_label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            # Find corresponding BFO class
            bfo_class = self.ontology.get_class_by_label(class_label)
            if bfo_class:
                results.append(ClassificationMatch(
                    class_uri=bfo_class.uri,
                    class_label=bfo_class.label,
                    confidence=min(score * 0.9, 0.95),  # Cap at 0.95 (rules not perfect)
                    source=self.get_source_name(),
                    metadata={'matched_keywords': list(scores.keys())}
                ))

        return results[:top_k]
