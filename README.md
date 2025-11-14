# BFO-Wikidata Classifier

**A multi-strategy, hierarchical classification system that maps Wikidata entities to Basic Formal Ontology (BFO) classes.**

**🔬 Research Preview** - This system is designed for research and experimentation only. While it can predict reasonable BFO classes in many cases, results may be unreliable due to embedding space limitations, BFO's abstract nature, and the inherent difficulty of mapping concrete entities to philosophical categories. Use with appropriate caution and validation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Classify an entity
python scripts/classify_entity.py Q2  # Earth

# 3. Use hierarchical classification
python scripts/classify_entity.py Q2 --hierarchical

# 4. Try different strategies
python scripts/classify_entity.py Q2 --strategy cascade
python scripts/classify_entity.py Q2 --preset research

# 5. Classify a random Wikidata entity
python scripts/classify_entity.py --random
```

---

## Why is BFO Classification Hard?

**BFO (Basic Formal Ontology) is an upper-level ontology with extremely abstract, philosophical classes:**

- **Entity** - "Anything that exists or has existed or will exist" ← Too general for any real classification
- **Continuant** - "An entity that exists in full at any time" ← Very abstract distinction
- **MaterialEntity** - "An independent continuant with matter" ← Still quite general

### The Core Problem

**BFO classes are so abstract that even "correct" classifications have very low semantic similarity:**

```python
# Marie Curie should be classified as Object (child of MaterialEntity)
Wikidata: "Marie Curie: Polish-French physicist and chemist"

# But the BFO definition is highly abstract
BFO Object: "(Elucidation) An object is a material entity which manifests
             causal unity & is of a type instances of which..."

Cosine Similarity: 0.18  # Very low! Even the "correct" class has poor similarity.

# The examples help somewhat, but not enough
BFO Object Examples: "An organism; a fish tank; a planet; a laptop; a valve..."
```

**The correct BFO class for Marie Curie is Object** (a person is a material object), but the philosophical definition shares almost no vocabulary with "physicist and chemist".

### Why Semantic Similarity Fails

1. **Abstract Definitions** - BFO definitions use philosophical language ("continuant", "occurrent", "specifically dependent") that has low semantic overlap with concrete Wikidata descriptions

2. **No Examples in Upper Classes** - Top-level BFO classes (Entity, Continuant, Occurrent) have no concrete examples, only abstract philosophical statements

3. **Low Similarity Scores** - Even good matches typically have cosine similarity ~0.2-0.4 (20-40%), not the 0.7-0.9 you'd expect for well-matched text

### The Workarounds

This system uses multiple strategies to cope with BFO's abstract nature:

1. **Wikidata Claim Rules** - Structural mappings (e.g., P31=Q5 "instance of human" → MaterialEntity) bypass semantic similarity entirely
2. **Hierarchical Classification** - Top-down traversal helps by comparing only siblings at each level
3. **Zero-Shot NLI** - Better at understanding abstract implications than pure semantic similarity
4. **Hybrid Strategies** - Combine multiple approaches to compensate for individual weaknesses
5. **SKOS Example Enrichment** - See [Semantic Embedding Strategy](#semantic-embedding-strategy) below

**Bottom line**: This is a research preview demonstrating techniques for classifying against abstract ontologies. The system is functional for prototyping and exploring BFO classifications.

### Example Output

**Flat Cascade Classification** (Earth Q2):
```bash
Top Matches:
1. three-dimensional spatial region (0.645) [semantic]
2. spatiotemporal region (0.610) [semantic]
3. site (0.596) [semantic]

CASCADE DECISION TRACE:
RULE_BASED:   [NO_RESULTS]
SEMANTIC:     three-dimensional spatial region (0.645 > 0.55 threshold) → [ACCEPTED]
```

**Hierarchical Classification** (Earth Q2):
```bash
HIERARCHICAL CLASSIFICATION PATH
Depth: 4 levels
Stop Reason: LEAF_NODE

Level 1: entity (conf: 1.000) (root)
  Level 2: occurrent (conf: 0.524) ->
    Level 3: spatiotemporal region (conf: 0.610) ->
      Level 4: spatiotemporal region (conf: 0.610) [LEAF]

Final Classification: spatiotemporal region (0.610)
```

Hierarchical mode successfully navigates the BFO tree from Entity → Occurrent → Spatiotemporal Region, producing a sensible classification for Earth as something that exists in both space and time.

---

## Table of Contents

- [Current Status](#current-status)
- [Installation](#installation)
- [Usage](#usage)
- [Classification Modes](#classification-modes)
- [Semantic Embedding Strategy](#semantic-embedding-strategy)
- [BFO Ontology Structure](#bfo-ontology-structure)
- [Known Issues](#known-issues)
- [Documentation](#documentation)

---

## Current Status

### ✅ Implemented & Working

- **Semantic Similarity Classifier** - SBERT embeddings + cosine similarity (low confidence scores expected)
- **Zero-Shot Classifier** - NLI-based, handles abstract reasoning better than semantic similarity
- **Rule-Based Classifier** - Wikidata P31 claim rules (only 1 rule: Q5→MaterialEntity)
- **Hybrid Strategies** - Cascade, ensemble, hybrid-confidence, tiered (all working)
- **Hierarchical Classification** - Top-down traversal from Entity through full depth
- **Wikidata Integration** - Live API + P31 "instance of" claim parsing
- **Official BFO Ontology** - Loads from ontologies/bfo-2020.ttl (36 core classes)
- **Model Registry** - 5 presets (production, research, resource_constrained, multilingual, ultra_lightweight)

### ⚠️ Partially Implemented

- **Rule-Based Classifier**
  - ✅ Wikidata claim rules (P31 "instance of") - HIGH confidence (0.95) when rules match
  - ⚠️ Keyword matching - basic stub, rarely matches
  - ❌ Advanced pattern matching not implemented

### ❌ Not Implemented

- **Fine-Tuned Classifier** - Complete stub, returns mock predictions only
- **Confidence Calibration** - Scores are not calibrated to 0-1 range
- **Multi-label Classification** - Entities can only map to one BFO class
- **Explanation Generation** - No explanations for why a class was chosen

---

## Installation

### Prerequisites

- Python 3.8+
- **No GPU required** - Models are small enough for CPU inference (~1-2GB RAM)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The system uses CPU-only inference. The BERT models (SBERT, zero-shot) are small enough (~80-400MB each) that GPU acceleration provides minimal benefit. Installation is simpler without CUDA/PyTorch GPU dependencies.

---

## Usage

### Command Line Flags

| Flag | Description | Example |
|------|-------------|---------|
| `ENTITY_ID` | Wikidata ID to classify | `Q7186` (Marie Curie) |
| `--random` | **Classify a random Wikidata entity (exploration mode)** | `--random` |
| `--strategy` | Hybrid strategy: `cascade`, `ensemble`, `hybrid_confidence`, `tiered` | `--strategy ensemble` |
| `--preset` | Model preset: `production`, `research`, `resource_constrained`, `multilingual`, `ultra_lightweight` | `--preset research` |
| `--hierarchical` | Use top-down hierarchical classification | `--hierarchical` |
| `--compare` | Compare all strategies on same entity | `--compare` |
| `--top-k` | Number of results to return (default: 3) | `--top-k 5` |
| `--sample` | Use sample entities instead of API | `--sample` |
| `--log` | Save results to log file | `--log` |

### Basic Classification

```bash
# Classify Marie Curie (Q7186)
python scripts/classify_entity.py Q7186

# Explore random Wikidata entities
python scripts/classify_entity.py --random

# Use different strategy
python scripts/classify_entity.py Q7186 --strategy ensemble

# Use different preset
python scripts/classify_entity.py Q7186 --preset research
```

### Hierarchical Classification

```bash
# Top-down classification from Entity
python scripts/classify_entity.py Q2 --hierarchical

# Try with different entities
python scripts/classify_entity.py Q7186 --hierarchical  # Marie Curie
python scripts/classify_entity.py Q42 --hierarchical     # Douglas Adams
```

**Example output (Earth Q2):**
```
HIERARCHICAL CLASSIFICATION PATH
Depth: 4 levels
Stop Reason: LEAF_NODE

Level 1: entity (conf: 1.000) (root)
  Level 2: occurrent (conf: 0.524) ->
    Level 3: spatiotemporal region (conf: 0.610) ->
      Level 4: spatiotemporal region (conf: 0.610) [LEAF]

Final Classification: spatiotemporal region (0.610)
```

Hierarchical mode navigates the full BFO tree and produces specific classifications.

### Compare Strategies

```bash
# Compare all strategies on one entity
python scripts/classify_entity.py Q7186 --compare
```

### Python API

```python
from src.models.ontology import BFOOntology
from src.models.wikidata import WikidataAPI
from src.classifiers.hybrid import HybridClassifier
from src.utils.model_registry import ModelRegistry
import yaml

# Setup
with open('configs/classification.yaml') as f:
    config = yaml.safe_load(f)

ontology = BFOOntology('ontologies/bfo-2020.ttl')  # Load from official BFO .ttl file
registry = ModelRegistry()
classifier = HybridClassifier(ontology, config, model_registry=registry)
api = WikidataAPI()

# Fetch entity
entity = api.fetch_entity("Q7186")  # Marie Curie

# Classify
result = classifier.classify(
    entity,
    strategy='cascade',
    top_k=3,
    hierarchical=True  # Use hierarchical mode
)

print(f"Class: {result.matches[0].class_label}")
print(f"Confidence: {result.matches[0].confidence:.3f}")  # Expect low!
print(f"Time: {result.processing_time_ms:.1f}ms")
```

---

## Classification Modes

### Flat Classification (Default)

Classifies across all 36 BFO classes simultaneously.

**Pros:**
- Fast - single classification pass
- Simple - no traversal logic

**Cons:**
- **Almost always returns "Entity"** - The most abstract class matches everything with decent similarity
- Doesn't leverage ontology hierarchy
- Low confidence scores (0.15-0.30 typical)

### Hierarchical Classification

Top-down classification starting from Entity, progressively narrowing to specific classes.

**How it works:**
1. Start at Entity (confidence: 1.0)
2. Get direct children from ontology (e.g., Continuant, Occurrent)
3. Classify among ONLY those children using semantic similarity
4. Pick best child and continue down
5. Stop when:
   - Reaches leaf node
   - Confidence < 0.50 (configurable)
   - Confidence drops > 0.15 from parent (configurable)
   - No good child match

**Pros:**
- Navigates full hierarchy (4-7 levels deep)
- More specific classifications than flat mode
- Leverages ontology structure
- Stops at appropriate level of specificity

**Cons:**
- Slower (multiple classification passes)
- Can get stuck in wrong branch early

**Example:** Earth (Q2) → Entity → Occurrent → Spatiotemporal Region

**Configuration:**
```yaml
# configs/classification.yaml
hierarchical:
  min_confidence: 0.50           # Minimum confidence to continue down hierarchy
  confidence_drop_threshold: 0.15  # Max drop allowed (depth 1+)
  # Note: Depth 0 (Entity→children) uses adaptive threshold of 0.50
  # to handle the large initial drop from root
```

---

## Semantic Embedding Strategy

The semantic similarity classifier addresses BFO's abstract terminology by enriching the embedding space with concrete examples from the ontology.

### SKOS Example Integration

BFO classes include `skos:example` annotations with concrete instances. These are concatenated with class definitions before embedding:

```python
# Example for BFO:Object class
Definition: "A material entity which manifests causal unity..."
Examples: "organism; fish tank; planet; laptop; valve; a grain of sand"

# Combined text for embedding:
"Object: A material entity which manifests causal unity...
 Examples: organism; fish tank; planet; laptop; valve; a grain of sand"
```

**Impact**: Including examples improves semantic matching because the embedding captures both the abstract philosophical definition AND concrete instances. This helps match real-world entities (like "Earth: third planet") to abstract BFO classes by their example similarity.

**Implementation**: See `BFOClass.get_text_for_embedding()` in [src/models/ontology.py](src/models/ontology.py)

**Limitations**: Not all BFO classes have examples (especially upper-level classes like Entity, Continuant, Occurrent), so this technique only partially addresses the abstraction problem.

---

## BFO Ontology Structure

The system loads the official BFO 2020 ontology from `ontologies/bfo-2020.ttl`. This contains 36 core BFO classes with proper hierarchy:

```
Entity (root)
├── Continuant (exists in full at any time)
│   ├── IndependentContinuant
│   │   ├── MaterialEntity
│   │   │   ├── Object (person, car, molecule)
│   │   │   ├── ObjectAggregate (population, collection)
│   │   │   └── FiatObjectPart (upper half of tree, Northern hemisphere)
│   │   └── ImmaterialEntity
│   │       ├── ContinuantFiatBoundary
│   │       │   ├── FiatPoint
│   │       │   ├── FiatLine
│   │       │   └── FiatSurface
│   │       └── Site
│   ├── SpecificallyDependentContinuant
│   │   ├── Quality
│   │   │   └── RelationalQuality
│   │   └── RealizableEntity
│   │       ├── Disposition
│   │       ├── Function
│   │       └── Role
│   ├── GenericallyDependentContinuant
│   └── SpatialRegion
│       ├── ThreeDimensionalSpatialRegion
│       ├── TwoDimensionalSpatialRegion
│       └── OneDimensionalSpatialRegion
└── Occurrent (unfolds in time)
    ├── Process
    │   ├── History
    │   └── ProcessBoundary
    ├── SpatiotemporalRegion
    └── TemporalRegion
        ├── OneDimensionalTemporalRegion
        │   └── TemporalInterval
        └── ZeroDimensionalTemporalRegion
            └── TemporalInstant
```

**Total**: 36 classes from official BFO 2020 .ttl file

---

## Configuration Presets

5 presets available via `--preset` flag:

| Preset | Description | Latency | Use Case |
|--------|-------------|---------|----------|
| **production** (default) | Speed priority | ~50-180ms | Production APIs |
| **research** | Quality priority | ~550ms | Offline analysis |
| **resource_constrained** | Minimal memory/compute | ~80ms | Edge devices |
| **multilingual** | 50+ languages | ~200ms | International data |
| **ultra_lightweight** | Extreme constraints | ~30ms | Mobile/embedded |

**Configuration:** See [configs/models.yaml](configs/models.yaml) for model selections per preset.

---

## Known Issues

### 1. Low Confidence Scores Are Normal

**Issue:** Semantic similarity typically returns confidence ~0.15-0.30 (15-30%) even for correct classifications.

**Why:** BFO definitions are abstract philosophical statements with low semantic overlap to concrete Wikidata descriptions. Cosine similarity between:
- `"Marie Curie: physicist and chemist"`
- `"MaterialEntity: An independent continuant that has some portion of matter..."`

...is naturally very low because the vocabulary doesn't overlap.

**Impact:**
- Cascade strategy rarely triggers early exit (threshold 0.55)
- System usually falls through to zero-shot classifier
- Users see surprisingly low confidence numbers

**This is expected behavior, not a bug.**

### 2. Flat Classification Returns "Entity"

**Issue:** Flat classification almost always returns "Entity" as the top match.

**Why:** "Entity" is defined as "anything that exists" - this matches everything! The definition is so broad that it has reasonable semantic similarity to any input.

**Workaround:** Use hierarchical classification (`--hierarchical`)

### 3. Zero-Shot NLI Misinterprets BFO Classes

**Issue:** Zero-shot classifier confuses everyday language meanings with BFO philosophical meanings.

**Example:** "Marie Curie: physicist and chemist" → classified as **Role** (0.735 confidence)
- **Why:** NLI model sees "physicist/chemist" and thinks "profession/job" (everyday meaning of "role")
- **BFO meaning:** Role = "a realizable entity that exists because there is some single bearer" (abstract philosophical concept)

**Impact:** Zero-shot performs poorly on concrete entities because BFO class labels ("role", "function", "disposition") have different meanings in common language vs. philosophy.

**Workaround:** Expand Wikidata claim rules (see below) to catch common cases before falling back to zero-shot.

### 4. Rule-Based Classifier Has Only 1 Rule

**Issue:** Rule-based classifier has only:
- 1 Wikidata claim rule: P31=Q5 ("instance of" human) → MaterialEntity
- ~10 keywords per class (basic stub, rarely matches)

**Impact:** Rules almost never match. System relies heavily on semantic/zero-shot.

**Workaround:** Add more Wikidata claim rules to [src/classifiers/rule_based.py](src/classifiers/rule_based.py):
```python
self.wikidata_rules = {
    'Q5': 'Object',          # human → Object
    'Q16521': 'Object',      # taxon → Object
    'Q1656682': 'Process',   # event → Process
    'Q11173': 'Quality',     # chemical compound → Quality
    # Add more...
}
```

### 5. Fine-Tuned Classifier Not Implemented

**Issue:** Fine-tuned classifier is a complete stub, returns mock predictions only.

**Impact:** Cannot be used for real classification.

**Future Fix:**
- Collect labeled training data (Wikidata entities → BFO classes)
- Implement training loop
- Fine-tune BERT/DeBERTa

### 6. Hierarchical Classification Confidence Normalization (FIXED)

**Issue:** Hierarchical classification was producing negative confidence scores and stopping too early.

**Root Cause:** Raw cosine similarity scores (range [-1, 1]) were being used directly as confidence values.

**Fix Applied:** Normalized similarity scores to [0, 1] range: `confidence = (similarity + 1.0) / 2.0`

**Example Results:**
```
Earth (Q2) Classification:
Level 1: entity (1.000) → START
Level 2: occurrent (0.524) → CONTINUE
Level 3: spatiotemporal region (0.610) → CONTINUE
Level 4: spatiotemporal region (0.610) → LEAF_NODE

Result: spatiotemporal region (confidence: 0.610)
```

**Status:** ✅ Fixed. Hierarchical classification now traverses 4-7 levels deep with proper confidence scores.

### 7. Hierarchical Classification Can Be Slow

**Issue:** Hierarchical mode requires multiple classification passes (one per level, up to 6-7 levels).

**Impact:** Can be 3-6x slower than flat classification.

**Workaround:**
- Use flat classification when speed is critical
- Use `ultra_lightweight` preset for hierarchical
- Adjust early stopping thresholds

### 8. No Confidence Calibration

**Issue:** Confidence scores are raw cosine similarity (0.3-0.7 range) or NLI scores, not calibrated probabilities.

**Impact:** Scores don't represent true certainty. A 0.25 similarity might be a correct match for BFO!

**Future Fix:** Implement temperature scaling or Platt scaling to map scores to 0-1 probability range.

---

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | This file - main documentation |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute getting started guide |
| [HIERARCHICAL_CLASSIFICATION.md](HIERARCHICAL_CLASSIFICATION.md) | Hierarchical mode guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture details |
| [DOCS.md](DOCS.md) | Documentation navigation index |
| [configs/models.yaml](configs/models.yaml) | Model registry + presets |
| [configs/classification.yaml](configs/classification.yaml) | Classifier thresholds |

---

## Future Work

### Immediate Priorities

1. **Expand Wikidata Claim Rules** - Map P31 values to specific BFO classes (not just Q5→MaterialEntity)
2. **Confidence Calibration** - Temperature scaling to get meaningful 0-1 probabilities

### Research Directions

- **Alternative Embeddings** - Try domain-adapted embeddings for philosophical/ontology text
- **Few-Shot Learning** - Use examples from BFO to guide classification
- **Hybrid Reasoning** - Combine semantic similarity with logical inference rules
- **Multi-Label Classification** - Allow entities to belong to multiple BFO classes
- **Explanation Generation** - Explain why a particular class was chosen

---

## References

- **BFO 2020**: https://basic-formal-ontology.org/
- **Wikidata**: https://www.wikidata.org/
- **Sentence-BERT**: https://www.sbert.net/
- **Hugging Face Transformers**: https://huggingface.co/transformers/

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Attribution

Tyler T. Procko for project orchestration, ideation, direction, iteration and refinement.

Claude (Anthropic) for source files and documentation.

---

## Acknowledgments

This project demonstrates the challenges of classifying concrete entities (Wikidata) against abstract upper ontologies (BFO). The low confidence scores and difficulties are inherent to this task, not implementation flaws. Future work on ontology alignment and semantic matching techniques could help bridge this gap.
