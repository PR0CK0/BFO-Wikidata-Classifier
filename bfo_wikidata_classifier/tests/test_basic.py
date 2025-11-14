"""
Basic tests for BFO-Wikidata classifier
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ontology import BFOOntology, BFOClass
from models.wikidata import WikidataEntity, create_sample_entities
from models.results import ClassificationMatch, ClassificationResult
from classifiers.rule_based import RuleBasedClassifier
from classifiers.semantic import SemanticClassifier
from utils.synthetic_data import generate_synthetic_examples


def get_ontology():
    """Helper to load ontology from .ttl file"""
    ontology_path = Path(__file__).parent.parent / "ontologies" / "bfo-2020.ttl"
    return BFOOntology(str(ontology_path))


def test_ontology_loading():
    """Test BFO ontology loading"""
    ontology = get_ontology()

    assert len(ontology.get_all_classes()) > 0
    assert ontology.get_class("bfo:MaterialEntity") is not None

    # Test hierarchy
    material_entity = ontology.get_class("bfo:MaterialEntity")
    ancestors = ontology.get_ancestors(material_entity.uri)
    assert "bfo:IndependentContinuant" in ancestors
    assert "bfo:Continuant" in ancestors

    print(f"✓ Ontology test passed: {len(ontology.get_all_classes())} classes loaded")


def test_wikidata_entity():
    """Test Wikidata entity model"""
    entity = WikidataEntity(
        id="Q7186",
        label="Marie Curie",
        description="Polish-French physicist",
        aliases=["Maria Curie"]
    )

    assert entity.get_text() == "Marie Curie: Polish-French physicist"
    assert entity.has_many_aliases() is False
    assert entity.has_short_description() is True

    print("✓ Wikidata entity test passed")


def test_rule_based_classifier():
    """Test rule-based classifier"""
    ontology = get_ontology()
    classifier = RuleBasedClassifier(ontology)

    # Test MaterialEntity
    entity = WikidataEntity("Q1", "Person", "A human being", [])
    matches = classifier.classify(entity, top_k=3)

    assert len(matches) > 0
    assert any("Material" in m.class_label or "Entity" in m.class_label for m in matches)

    # Test Process
    entity = WikidataEntity("Q2", "War", "Armed conflict or event", [])
    matches = classifier.classify(entity, top_k=3)

    assert len(matches) > 0
    assert any("Process" in m.class_label for m in matches)

    print("✓ Rule-based classifier test passed")


def test_semantic_classifier():
    """Test semantic similarity classifier"""
    ontology = get_ontology()
    classifier = SemanticClassifier(ontology, config={})

    entity = WikidataEntity("Q7186", "Marie Curie", "Polish-French physicist", [])
    matches = classifier.classify(entity, top_k=3)

    assert len(matches) > 0
    assert matches[0].confidence > 0.0
    assert matches[0].source == 'semantic'

    print(f"✓ Semantic classifier test passed: top match = {matches[0].class_label}")


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    ontology = get_ontology()
    examples = generate_synthetic_examples(ontology)

    assert len(examples) > 0
    entity, ground_truth = examples[0]
    assert isinstance(entity, WikidataEntity)
    assert isinstance(ground_truth, str)

    print(f"✓ Synthetic data test passed: {len(examples)} examples generated")


def test_classification_result():
    """Test classification result model"""
    match1 = ClassificationMatch(
        class_uri="bfo:MaterialEntity",
        class_label="MaterialEntity",
        confidence=0.85,
        source="semantic"
    )

    match2 = ClassificationMatch(
        class_uri="bfo:IndependentContinuant",
        class_label="IndependentContinuant",
        confidence=0.75,
        source="semantic"
    )

    result = ClassificationResult(
        entity_id="Q7186",
        entity_label="Marie Curie",
        matches=[match1, match2],
        strategy="cascade",
        processing_time_ms=50.0
    )

    assert result.get_top_match() == match1
    assert result.get_top_match().confidence == 0.85
    assert len(result.get_all_classes()) == 2

    result_dict = result.to_dict()
    assert result_dict['entity_id'] == "Q7186"

    print("✓ Classification result test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("RUNNING BASIC TESTS")
    print("=" * 80)

    tests = [
        test_ontology_loading,
        test_wikidata_entity,
        test_rule_based_classifier,
        test_semantic_classifier,
        test_synthetic_data_generation,
        test_classification_result,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
