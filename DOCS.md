# Documentation Index

Quick guide to navigating the BFO-Wikidata Classifier documentation.

---

## Start Here

**New User?** Start with [QUICKSTART.md](QUICKSTART.md) for a 5-minute getting started guide.

**Want Details?** See [README.md](README.md) for comprehensive documentation.

---

## Documentation Files

### User Documentation

| File | Purpose | Audience |
|------|---------|----------|
| **[README.md](README.md)** | Main documentation - installation, usage, architecture, known issues | Everyone |
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute getting started guide | New users |
| **[HIERARCHICAL_CLASSIFICATION.md](HIERARCHICAL_CLASSIFICATION.md)** | Detailed guide to hierarchical classification mode | Users wanting top-down classification |

### Developer Documentation

| File | Purpose | Audience |
|------|---------|----------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Technical architecture, implementation details, extension points | Developers |
| **[configs/models.yaml](configs/models.yaml)** | Model registry and presets configuration | Developers |
| **[configs/classification.yaml](configs/classification.yaml)** | Classifier thresholds and settings | Developers |

---

## Quick Navigation

**Want to...**

- **Get started quickly?** → [QUICKSTART.md](QUICKSTART.md)
- **Understand what this does?** → [README.md#overview](README.md#overview)
- **Install with GPU support?** → [README.md#installation](README.md#installation)
- **Try hierarchical classification?** → [HIERARCHICAL_CLASSIFICATION.md](HIERARCHICAL_CLASSIFICATION.md)
- **Use different model presets?** → [README.md#configuration-presets](README.md#configuration-presets)
- **Understand known issues?** → [README.md#known-issues](README.md#known-issues)
- **See architecture details?** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Add Wikidata claim rules?** → [src/classifiers/rule_based.py](src/classifiers/rule_based.py)
- **Change model selection?** → [configs/models.yaml](configs/models.yaml)
- **Tune thresholds?** → [configs/classification.yaml](configs/classification.yaml)

---

## Documentation Status

### What's Documented

✅ Installation with GPU support
✅ Basic and hierarchical classification
✅ Model presets and configuration
✅ Known issues and limitations
✅ Python API usage
✅ Architecture and design

### What's Not Yet Documented

❌ Fine-tuned classifier (not implemented)
❌ Training data collection process
❌ Advanced rule pattern matching
❌ Multi-label classification
❌ Confidence calibration techniques

---

## Contributing to Docs

Found an issue or want to improve the docs?

1. Check existing docs for coverage
2. Update the relevant file(s)
3. Add entry to this index if needed
4. Keep docs honest about what's implemented vs. stub

---

## File Structure

```
bfo_wikidata_classifier/
├── README.md                          ← Main documentation
├── QUICKSTART.md                      ← 5-minute guide
├── DOCS.md                            ← This file
├── HIERARCHICAL_CLASSIFICATION.md     ← Hierarchical mode guide
├── ARCHITECTURE.md                    ← Technical details
├── configs/
│   ├── models.yaml                    ← Model registry
│   └── classification.yaml            ← Classifier config
├── src/                               ← Source code
└── scripts/                           ← Demo scripts
```

---

**Questions?** Check [README.md](README.md) or review the code comments.
