# Hierarchical Classification Guide

> **Note:** This is a detailed guide for hierarchical classification. For general usage, see [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md).

---

## Overview

The BFO-Wikidata classifier now supports **two classification modes**:

1. **Flat Classification** (default): Classifies across all BFO classes at once
2. **Hierarchical Classification** (new): Top-down classification starting from Entity

## The Problem with Flat Classification

When using flat classification, the system often returns overly general results:

```bash
$ python scripts/classify_entity.py Q7186 --strategy cascade

CLASSIFICATION RESULTS
================================================================================
Entity: Marie Curie (Q7186)
Description: Polish-French physicist and chemist

Top Match:
1. Entity (confidence: 0.95)  ← Too obvious! Not useful!
```

**Why?** Because "Entity" is the most general class and almost always has high similarity to any input.

## Hierarchical Classification Solution

Hierarchical classification progressively narrows down from general to specific:

```bash
$ python scripts/classify_entity.py Q7186 --strategy cascade --hierarchical

HIERARCHICAL CLASSIFICATION PATH
================================================================================
Depth: 5 levels
Stop Reason: LEAF_NODE

Level 1: Entity (conf: 1.000) →
  Level 2: Continuant (conf: 0.820) →
    Level 3: IndependentContinuant (conf: 0.750) →
      Level 4: MaterialEntity (conf: 0.680) →
        Level 5: Object (conf: 0.598) ✓ (leaf node)

FINAL CLASSIFICATION: Object (0.598)
```

## How It Works

### Step-by-Step Process

1. **Start at Root**: Begin with Entity (always confident: 1.0)

2. **Classify Among Children**: At each level, classify among ONLY the direct children
   - Entity's children: Continuant vs Occurrent
   - Pick best match: Continuant (0.82)

3. **Continue Down**: Move to Continuant, classify among ITS children
   - Continuant's children: IndependentContinuant, DependentContinuant, SpatialRegion
   - Pick best match: IndependentContinuant (0.75)

4. **Stop When**:
   - **Leaf Node**: No more children (e.g., MaterialEntity)
   - **Low Confidence**: Child confidence < 0.50 (configurable)
   - **Confidence Drop**: Parent → Child drop > 0.15 (configurable)
   - **No Match**: No good child match found

### Example: Marie Curie (Physicist)

```
Input: "Marie Curie: Polish-French physicist and chemist"

Entity (1.00)
├─ Continuant (0.82) ✓ Selected
│  ├─ IndependentContinuant (0.75) ✓ Selected
│  │  ├─ MaterialEntity (0.68) ✓ Selected (LEAF)
│  │  └─ ImmaterialEntity (0.32)
│  ├─ DependentContinuant (0.45)
│  └─ SpatialRegion (0.18)
└─ Occurrent (0.18)
   ├─ Process (0.15)
   └─ TemporalRegion (0.03)

Result: MaterialEntity (0.68) - Correct!
```

### Example: Running (Process)

```
Input: "Running: the act of moving rapidly on foot"

Entity (1.00)
├─ Continuant (0.35)
└─ Occurrent (0.82) ✓ Selected
   ├─ Process (0.75) ✓ Selected (LEAF)
   └─ TemporalRegion (0.25)

Result: Process (0.75) - Correct!
```

### Example: Color (Quality)

```
Input: "Red: a color at the end of the visible spectrum"

Entity (1.00)
├─ Continuant (0.88) ✓ Selected
│  ├─ IndependentContinuant (0.45)
│  ├─ DependentContinuant (0.72) ✓ Selected
│  │  ├─ Quality (0.68) ✓ Selected (LEAF)
│  │  └─ Role (0.32)
│  └─ SpatialRegion (0.20)
└─ Occurrent (0.12)

Result: Quality (0.68) - Correct!
```

## Configuration

Edit `configs/classification.yaml`:

```yaml
# Hierarchical classification settings
hierarchical:
  min_confidence: 0.50  # Minimum confidence to continue down
  confidence_drop_threshold: 0.15  # Max allowed confidence drop
```

### Tuning Parameters

- **`min_confidence`**: Lower = go deeper (more specific but risky)
  - 0.30: Aggressive (go deep even with low confidence)
  - 0.50: Balanced (default)
  - 0.70: Conservative (stop early, return general classes)

- **`confidence_drop_threshold`**: Lower = stricter (stop on small drops)
  - 0.10: Strict (stop if child confidence drops much)
  - 0.15: Balanced (default)
  - 0.25: Permissive (allow larger confidence drops)

## Usage

### Basic Usage

```bash
# Flat classification (default)
python scripts/classify_entity.py Q7186

# Hierarchical classification
python scripts/classify_entity.py Q7186 --hierarchical

# Random entity with hierarchical
python scripts/classify_entity.py --random --hierarchical

# With strategy selection
python scripts/classify_entity.py Q7186 --hierarchical --strategy ensemble
```

### With Logging

```bash
# Log hierarchical classification
python scripts/classify_entity.py --random --hierarchical --log

# The log file will contain the full hierarchical path
cat logs/20250113_143022_Q7186.yaml
```

### Example Log Output

```yaml
timestamp: '2025-01-13T14:30:22.123456'
entity:
  id: Q7186
  label: Marie Curie
  description: Polish-French physicist and chemist
classification:
  strategy: cascade_hierarchical
  processing_time_ms: 247.5
  top_matches:
    - class_uri: http://purl.obolibrary.org/obo/BFO_0000040
      class_label: MaterialEntity
      confidence: 0.68
      source: hierarchical_cascade
      metadata:
        hierarchical_path:
          - class_label: Entity
            confidence: 1.0
            decision: CONTINUE
          - class_label: Continuant
            confidence: 0.82
            decision: CONTINUE
          - class_label: IndependentContinuant
            confidence: 0.75
            decision: CONTINUE
          - class_label: MaterialEntity
            confidence: 0.68
            decision: LEAF_NODE
        stop_reason: LEAF_NODE
        depth: 4
```

## Stop Reasons Explained

### ✓ LEAF_NODE
- Reached a class with no children
- This is the most specific classification possible
- **Action**: Return this class as final result

### ⚠ LOW_CONFIDENCE
- Child confidence below `min_confidence` threshold
- Not confident enough to go deeper
- **Action**: Return parent class (more general but confident)

Example:
```
Level 3: IndependentContinuant (0.75) ⚠ (low confidence)
        ↳ Tried: MaterialEntity (0.40)
```

### ⚠ CONFIDENCE_DROP
- Confidence dropped too much from parent to child
- Drop exceeds `confidence_drop_threshold`
- **Action**: Return parent class (more confident)

Example:
```
Level 2: Continuant (0.82) ⚠ (confidence drop)
        ↳ Tried: IndependentContinuant (0.55)
        ↳ Drop: 0.27
```

### ✗ NO_CHILD_MATCH
- No children matched well enough
- All children have very low scores
- **Action**: Return parent class

## Comparison: Flat vs Hierarchical

| Aspect | Flat Classification | Hierarchical Classification |
|--------|---------------------|----------------------------|
| **Result** | Often too general (Entity) | Progressively specific |
| **Speed** | Fast (single pass) | Slower (multiple levels) |
| **Accuracy** | Lower (broad matching) | Higher (focused matching) |
| **Confidence** | Often overconfident | More realistic |
| **Use Case** | Quick filtering | Precise classification |

## When to Use Each Mode

### Use Flat Classification When:
- You need very fast results
- You want to see all possible classes
- You're doing exploratory analysis
- You want parent class inference

### Use Hierarchical Classification When:
- You need precise, specific classifications
- You want to avoid overly general results
- You're building a production system
- You want to understand the reasoning path

## Advanced: Strategy Comparison

All strategies work with hierarchical mode:

```bash
# Compare strategies in hierarchical mode
python scripts/classify_entity.py Q7186 --hierarchical --compare
```

Output:
```
CASCADE_HIERARCHICAL (248ms):
  1. MaterialEntity (0.68) [hierarchical_cascade]

ENSEMBLE_HIERARCHICAL (255ms):
  1. MaterialEntity (0.72) [hierarchical_ensemble]

HYBRID_CONFIDENCE_HIERARCHICAL (180ms):
  1. MaterialEntity (0.70) [hierarchical_hybrid_confidence]

TIERED_HIERARCHICAL (190ms):
  1. MaterialEntity (0.69) [hierarchical_tiered]
```

## Implementation Details

### Algorithm

```python
def hierarchical_classify(entity, strategy):
    current_node = Entity  # Start at root
    current_confidence = 1.0
    path = []

    while current_node:
        children = get_children(current_node)

        if not children:
            # LEAF_NODE
            path.append(current_node)
            break

        # Classify among children only
        child_results = classify_among(entity, children, strategy)

        if not child_results:
            # NO_CHILD_MATCH
            path.append(current_node)
            break

        best_child = child_results[0]

        if best_child.confidence < min_confidence:
            # LOW_CONFIDENCE
            path.append(current_node)
            break

        if current_confidence - best_child.confidence > confidence_drop_threshold:
            # CONFIDENCE_DROP
            path.append(current_node)
            break

        # Continue down
        path.append(best_child)
        current_node = best_child
        current_confidence = best_child.confidence

    return path[-1]  # Final classification
```

### Key Design Decisions

1. **Start with 1.0 confidence at root**: Entity always matches
2. **Filter after classification**: Run full classification, then filter to candidates
3. **Track full path**: Store reasoning for debugging and logging
4. **Multiple stop conditions**: Prevent over-classification
5. **Strategy-agnostic**: Works with cascade, ensemble, hybrid, tiered

## Troubleshooting

### Issue: Always stops at Entity
```
Level 1: Entity (1.000) ✗ (no child match)
```

**Solution**: Your semantic model might not be loaded correctly. Check:
```bash
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

### Issue: Goes too deep, low confidence
```
Level 5: SpecificClass (0.25) ⚠ (low confidence)
```

**Solution**: Increase `min_confidence` in config:
```yaml
hierarchical:
  min_confidence: 0.60  # Was 0.50
```

### Issue: Stops too early
```
Level 2: Continuant (0.82) ⚠ (confidence drop)
```

**Solution**: Increase `confidence_drop_threshold`:
```yaml
hierarchical:
  confidence_drop_threshold: 0.25  # Was 0.15
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test with samples**: `python scripts/classify_entity.py --sample Marie_Curie --hierarchical`
3. **Try random entities**: `python scripts/classify_entity.py --random --hierarchical`
4. **Tune thresholds**: Edit `configs/classification.yaml` based on results
5. **Log everything**: Use `--log` to track performance and patterns

## Performance Notes

- **Flat**: ~50ms (single classification pass)
- **Hierarchical**: ~150-300ms (multiple passes, depends on depth)
- **Memory**: Same as flat (no additional models loaded)
- **Depth**: Typically 2-5 levels for BFO ontology

The increased time provides more specific classifications than flat mode.

**Note**: The system uses SKOS examples from BFO class definitions (e.g., "organism; fish tank; planet; laptop") embedded alongside elucidations to improve semantic matching against abstract philosophical terms.
