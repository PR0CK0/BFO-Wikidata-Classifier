"""
Semantic similarity classifier using sentence transformers

Primary classification approach using SBERT embeddings and cosine similarity.
"""

import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from classifiers.base import BaseClassifier
from models.wikidata import WikidataEntity
from models.results import ClassificationMatch


class SemanticClassifier(BaseClassifier):
    """Semantic similarity classifier using SBERT"""

    def __init__(self, ontology, config: dict = None, model: Optional[SentenceTransformer] = None):
        """
        Initialize semantic classifier

        Args:
            ontology: BFO ontology
            config: Configuration dict
            model: Pre-loaded SentenceTransformer (or None to load default)
        """
        super().__init__(ontology, config)

        # Load or use provided model
        if model is None:
            model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
            print(f"Loading semantic model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            self.model = model

        # Pre-compute ontology embeddings
        self.class_embeddings = None
        self.class_uris = []
        self.index_ontology()

    def index_ontology(self):
        """Pre-compute embeddings for all BFO classes"""
        print("Indexing BFO ontology...")

        # Get all classes
        classes = self.ontology.get_all_classes()

        if not classes:
            raise ValueError("Ontology has no classes to index")

        # Extract texts for embedding
        texts = [cls.get_text_for_embedding() for cls in classes]
        self.class_uris = [cls.uri for cls in classes]

        # Compute embeddings
        self.class_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        print(f"Indexed {len(classes)} classes")

    def _normalize_similarity(self, raw_similarity: float) -> float:
        """
        Normalize cosine similarity (-1 to 1) to confidence (0 to 1)

        Uses a simple linear transformation: (sim + 1) / 2
        This maps:
          -1.0 -> 0.0 (completely opposite)
           0.0 -> 0.5 (orthogonal/unrelated)
           1.0 -> 1.0 (identical)

        Args:
            raw_similarity: Raw cosine similarity score

        Returns:
            Normalized confidence score in [0, 1]
        """
        return (raw_similarity + 1.0) / 2.0

    def classify(self, entity: WikidataEntity, top_k: int = 3) -> List[ClassificationMatch]:
        """
        Classify using semantic similarity

        Args:
            entity: Wikidata entity
            top_k: Number of results

        Returns:
            List of matches
        """
        if self.class_embeddings is None:
            raise RuntimeError("Ontology not indexed. Call index_ontology() first.")

        # Encode entity text
        entity_text = entity.get_text()
        entity_emb = self.model.encode(entity_text, convert_to_numpy=True)

        # Compute similarities
        similarities = cosine_similarity(
            [entity_emb],
            self.class_embeddings
        )[0]

        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        min_similarity = self.config.get('min_similarity', 0.3)

        for idx in top_indices:
            uri = self.class_uris[idx]
            bfo_class = self.ontology.get_class(uri)
            raw_similarity = float(similarities[idx])

            # Normalize to 0-1 range
            confidence = self._normalize_similarity(raw_similarity)

            if bfo_class:
                # Always include results, but log if below threshold
                if raw_similarity < min_similarity:
                    # Still add it but mark as low confidence
                    pass

                results.append(ClassificationMatch(
                    class_uri=bfo_class.uri,
                    class_label=bfo_class.label,
                    confidence=confidence,
                    source=self.get_source_name(),
                    metadata={
                        'raw_similarity': raw_similarity,
                        'entity_text': entity_text,
                        'below_threshold': raw_similarity < min_similarity
                    }
                ))

        return results
