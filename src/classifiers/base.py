"""
Base classifier interface
"""

from abc import ABC, abstractmethod
from typing import List

from models.ontology import BFOOntology
from models.wikidata import WikidataEntity
from models.results import ClassificationMatch


class BaseClassifier(ABC):
    """Abstract base class for all classifiers"""

    def __init__(self, ontology: BFOOntology, config: dict = None):
        """
        Initialize classifier

        Args:
            ontology: BFO ontology
            config: Classifier-specific configuration
        """
        self.ontology = ontology
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def classify(self, entity: WikidataEntity, top_k: int = 3) -> List[ClassificationMatch]:
        """
        Classify entity to BFO classes

        Args:
            entity: Wikidata entity to classify
            top_k: Number of top matches to return

        Returns:
            List of ClassificationMatch objects
        """
        pass

    def get_source_name(self) -> str:
        """Get classifier source name for results"""
        return self.name.replace('Classifier', '').lower()
