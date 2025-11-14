"""
Synthetic labeled data generator

Generates Wikidata-like entities with ground truth BFO classifications
for evaluation purposes.
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path

from models.wikidata import WikidataEntity
from models.ontology import BFOOntology


def generate_synthetic_examples(ontology: BFOOntology) -> List[Tuple[WikidataEntity, str]]:
    """
    Generate synthetic labeled examples

    Returns:
        List of (WikidataEntity, ground_truth_uri) tuples
    """
    examples = []

    # MaterialEntity examples (people, objects, organisms)
    material_entity_uri = None
    for cls in ontology.get_all_classes():
        if cls.label == "MaterialEntity":
            material_entity_uri = cls.uri
            break

    if material_entity_uri:
        examples.extend([
            (WikidataEntity("Q_SYNTH_1", "Marie Curie",
                           "Polish-French physicist and chemist (1867-1934)",
                           ["Maria Skłodowska-Curie"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_2", "Albert Einstein",
                           "German-born theoretical physicist (1879-1955)",
                           ["Einstein"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_3", "DNA",
                           "molecule that carries genetic information",
                           ["deoxyribonucleic acid"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_4", "Human",
                           "common name of Homo sapiens",
                           ["person", "human being"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_5", "Protein",
                           "biological molecule consisting of amino acid chains",
                           ["proteins"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_6", "Enzyme",
                           "biological molecule that catalyzes chemical reactions",
                           ["catalyst", "biocatalyst"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_7", "Eiffel Tower",
                           "iron lattice tower in Paris, France",
                           ["La Tour Eiffel"]), material_entity_uri),
            (WikidataEntity("Q_SYNTH_8", "Computer",
                           "programmable electronic device for data processing",
                           ["computing machine"]), material_entity_uri),
        ])

    # Process examples (events, activities)
    process_uri = None
    for cls in ontology.get_all_classes():
        if cls.label == "Process":
            process_uri = cls.uri
            break

    if process_uri:
        examples.extend([
            (WikidataEntity("Q_SYNTH_20", "World War II",
                           "global war from 1939 to 1945",
                           ["WW2", "WWII"]), process_uri),
            (WikidataEntity("Q_SYNTH_21", "French Revolution",
                           "period of radical social and political change in France",
                           ["Revolution"]), process_uri),
            (WikidataEntity("Q_SYNTH_22", "Cell Division",
                           "process by which a cell divides into two daughter cells",
                           ["mitosis", "cytokinesis"]), process_uri),
            (WikidataEntity("Q_SYNTH_23", "Photosynthesis",
                           "process used by plants to convert light into energy",
                           []), process_uri),
            (WikidataEntity("Q_SYNTH_24", "Olympic Games",
                           "international multi-sport event",
                           ["Olympics"]), process_uri),
            (WikidataEntity("Q_SYNTH_25", "Industrial Revolution",
                           "period of major industrialization and innovation",
                           []), process_uri),
            (WikidataEntity("Q_SYNTH_26", "Evolution",
                           "change in heritable characteristics over generations",
                           ["biological evolution"]), process_uri),
        ])

    # Quality examples (properties, characteristics)
    quality_uri = None
    for cls in ontology.get_all_classes():
        if cls.label == "Quality":
            quality_uri = cls.uri
            break

    if quality_uri:
        examples.extend([
            (WikidataEntity("Q_SYNTH_40", "Blue",
                           "color between violet and cyan on visible spectrum",
                           ["blue color"]), quality_uri),
            (WikidataEntity("Q_SYNTH_41", "Temperature",
                           "physical quantity expressing hot and cold",
                           ["thermal energy"]), quality_uri),
            (WikidataEntity("Q_SYNTH_42", "Mass",
                           "physical property of matter related to inertia",
                           ["weight"]), quality_uri),
            (WikidataEntity("Q_SYNTH_43", "Color",
                           "visual perception based on electromagnetic spectrum",
                           ["colour"]), quality_uri),
            (WikidataEntity("Q_SYNTH_44", "Shape",
                           "external form or outline of an object",
                           ["geometry"]), quality_uri),
            (WikidataEntity("Q_SYNTH_45", "Density",
                           "mass per unit volume",
                           []), quality_uri),
        ])

    # Role examples (functions, capacities)
    role_uri = None
    for cls in ontology.get_all_classes():
        if cls.label == "Role":
            role_uri = cls.uri
            break

    if role_uri:
        examples.extend([
            (WikidataEntity("Q_SYNTH_60", "Teacher",
                           "person who helps others to acquire knowledge",
                           ["educator", "instructor"]), role_uri),
            (WikidataEntity("Q_SYNTH_61", "Catalyst",
                           "substance that increases rate of chemical reaction",
                           ["catalytic agent"]), role_uri),
            (WikidataEntity("Q_SYNTH_62", "President",
                           "leader of a country or organization",
                           ["chief executive"]), role_uri),
            (WikidataEntity("Q_SYNTH_63", "Function",
                           "activity or purpose natural to a thing",
                           ["role", "purpose"]), role_uri),
        ])

    # SpatialRegion examples (locations, places)
    spatial_uri = None
    for cls in ontology.get_all_classes():
        if cls.label == "SpatialRegion":
            spatial_uri = cls.uri
            break

    if spatial_uri:
        examples.extend([
            (WikidataEntity("Q_SYNTH_80", "Europe",
                           "continent located in Northern Hemisphere",
                           []), spatial_uri),
            (WikidataEntity("Q_SYNTH_81", "Pacific Ocean",
                           "largest ocean on Earth",
                           []), spatial_uri),
            (WikidataEntity("Q_SYNTH_82", "Sahara",
                           "hot desert in Africa",
                           ["Sahara Desert"]), spatial_uri),
            (WikidataEntity("Q_SYNTH_83", "North Pole",
                           "northernmost point on Earth",
                           []), spatial_uri),
        ])

    return examples


def save_synthetic_data(examples: List[Tuple[WikidataEntity, str]], output_path: str):
    """
    Save synthetic examples to JSON file

    Args:
        examples: List of (entity, ground_truth_uri) tuples
        output_path: Path to output JSON file
    """
    data = []
    for entity, ground_truth_uri in examples:
        data.append({
            'entity': entity.to_dict(),
            'ground_truth_uri': ground_truth_uri
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} synthetic examples to {output_path}")


def load_synthetic_data(input_path: str) -> List[Tuple[WikidataEntity, str]]:
    """
    Load synthetic examples from JSON file

    Args:
        input_path: Path to JSON file

    Returns:
        List of (entity, ground_truth_uri) tuples
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data:
        entity = WikidataEntity.from_dict(item['entity'])
        ground_truth_uri = item['ground_truth_uri']
        examples.append((entity, ground_truth_uri))

    return examples


if __name__ == "__main__":
    from ..models.ontology import BFOOntology
    from pathlib import Path

    # Generate examples
    ontology_path = Path(__file__).parent.parent.parent / "ontologies" / "bfo-2020.ttl"
    ontology = BFOOntology(str(ontology_path))
    examples = generate_synthetic_examples(ontology)

    print(f"Generated {len(examples)} synthetic examples:")
    for entity, uri in examples[:5]:
        print(f"  {entity.label} -> {uri}")

    # Save to file
    output_path = "data/synthetic_labels.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_synthetic_data(examples, output_path)
