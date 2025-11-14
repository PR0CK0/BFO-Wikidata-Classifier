"""
Model Registry for Managing Multiple Classification Models

Provides centralized configuration and loading of:
- Semantic similarity models (sentence-transformers)
- Zero-shot classification models (transformers)
- Base models for fine-tuning (BERT, RoBERTa, etc.)
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, List, Any
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch


class ModelRegistry:
    """Registry for selecting and managing different classification models"""

    def __init__(self, config_path: Optional[str] = None, device: Optional[torch.device] = None):
        """
        Initialize model registry

        Args:
            config_path: Path to models.yaml config file
            device: torch.device to use for model inference (defaults to auto-detect)
        """
        if config_path is None:
            # Default to configs/models.yaml relative to project root
            config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Convert to device_id for transformers pipeline (-1 for CPU, 0+ for GPU)
        self.device_id = -1 if self.device.type == 'cpu' else self.device.index or 0

        # Cache loaded models to avoid reloading
        self._semantic_cache: Dict[str, SentenceTransformer] = {}
        self._zeroshot_cache: Dict[str, Any] = {}

    def get_semantic_model(self, name: Optional[str] = None) -> SentenceTransformer:
        """
        Get semantic similarity model (sentence-transformers)

        Args:
            name: Model name (e.g., "all-MiniLM-L6-v2"). If None, uses default.

        Returns:
            SentenceTransformer model
        """
        if name is None:
            name = self.config['semantic_models']['default']

        # Check cache
        if name not in self._semantic_cache:
            print(f"Loading semantic model: {name} on {self.device}")
            model = SentenceTransformer(name)
            model = model.to(self.device)
            self._semantic_cache[name] = model

        return self._semantic_cache[name]

    def get_zeroshot_model(self, name: Optional[str] = None):
        """
        Get zero-shot classification model (transformers pipeline)

        Args:
            name: Model name (e.g., "facebook/bart-large-mnli"). If None, uses default.

        Returns:
            Transformers zero-shot-classification pipeline
        """
        if name is None:
            name = self.config['zero_shot_models']['default']

        # Check cache
        if name not in self._zeroshot_cache:
            print(f"Loading zero-shot model: {name} on {self.device}")
            self._zeroshot_cache[name] = pipeline(
                "zero-shot-classification",
                model=name,
                device=self.device_id
            )

        return self._zeroshot_cache[name]

    def get_finetuned_base_model(self, name: Optional[str] = None) -> str:
        """
        Get base model name for fine-tuning

        Args:
            name: Model name (e.g., "distilbert-base-uncased"). If None, uses default.

        Returns:
            Model name string (for use with transformers.AutoModel)
        """
        if name is None:
            name = self.config['base_models_for_finetuning']['default']

        return name

    def list_available_models(self, model_type: str) -> List[Dict[str, Any]]:
        """
        List all available models of a given type with metadata

        Args:
            model_type: One of "semantic", "zeroshot", "finetuned"

        Returns:
            List of model info dictionaries
        """
        type_map = {
            'semantic': 'semantic_models',
            'zeroshot': 'zero_shot_models',
            'finetuned': 'base_models_for_finetuning'
        }

        if model_type not in type_map:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(type_map.keys())}")

        return self.config[type_map[model_type]]['options']

    def get_model_info(self, model_type: str, name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model

        Args:
            model_type: One of "semantic", "zeroshot", "finetuned"
            name: Model name

        Returns:
            Model info dictionary
        """
        models = self.list_available_models(model_type)
        for model_info in models:
            if model_info['name'] == name:
                return model_info

        raise ValueError(f"Model {name} not found in {model_type} models")

    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """
        Get preset configuration (production, research, etc.)

        Args:
            preset_name: One of "production", "research", "resource_constrained", "multilingual"

        Returns:
            Preset configuration dictionary
        """
        if preset_name not in self.config['presets']:
            raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(self.config['presets'].keys())}")

        return self.config['presets'][preset_name]

    def list_presets(self) -> List[str]:
        """List all available preset configurations"""
        return list(self.config['presets'].keys())

    def compare_models(self, model_type: str) -> str:
        """
        Generate comparison table for all models of a type

        Args:
            model_type: One of "semantic", "zeroshot", "finetuned"

        Returns:
            Formatted comparison table
        """
        models = self.list_available_models(model_type)

        # Build table
        lines = []
        lines.append(f"\n{model_type.upper()} MODELS COMPARISON")
        lines.append("=" * 80)

        for model in models:
            lines.append(f"\n{model['name']}")
            lines.append(f"  Description: {model['description']}")
            lines.append(f"  Size: {model['size']}")
            lines.append(f"  Speed: {model['speed']}")
            lines.append(f"  Quality: {model['quality']}")

            # Add type-specific metrics
            if model_type == 'semantic' and 'benchmark_sts' in model:
                lines.append(f"  STS Score: {model['benchmark_sts']}")
                lines.append(f"  Speed: {model['sentences_per_sec']} sent/sec")
            elif model_type == 'zeroshot' and 'mnli_accuracy' in model:
                lines.append(f"  MNLI Accuracy: {model['mnli_accuracy']}%")
                lines.append(f"  Latency: {model['avg_latency_ms']}ms")
            elif model_type == 'finetuned' and 'glue_score' in model:
                lines.append(f"  GLUE Score: {model['glue_score']}")
                lines.append(f"  Parameters: {model['params']}")

        return "\n".join(lines)


def print_model_justifications():
    """Print detailed justifications for default model choices"""

    justifications = """
======================================================================================
MODEL JUSTIFICATIONS
======================================================================================

1. SEMANTIC SIMILARITY: all-MiniLM-L6-v2
--------------------------------------------------------------------------------------
Why chosen:
  ✅ Speed: 80MB, ~50ms/sentence on CPU (14,000 sent/sec)
  ✅ Quality: 95% of BERT-base performance (STS: 68.06)
  ✅ Efficiency: Best speed/quality tradeoff
  ✅ Universal: Works well across all domains
  ✅ Proven: 50M+ downloads on HuggingFace

Trade-off:
  - 2.3% less accurate than all-mpnet-base-v2, but 5x faster
  - Perfect for production and demos

When to use alternatives:
  - all-mpnet-base-v2: Need highest accuracy, have GPU
  - paraphrase-multilingual-*: Multi-language support needed

Benchmarks:
  all-MiniLM-L6-v2:    STS: 68.06  Speed: 14,000 sent/sec  Size: 80MB
  all-mpnet-base-v2:   STS: 69.57  Speed: 2,800 sent/sec   Size: 420MB
  bert-base-uncased:   STS: 65.50  Speed: 1,000 sent/sec   Size: 440MB

2. ZERO-SHOT CLASSIFICATION: facebook/bart-large-mnli
--------------------------------------------------------------------------------------
Why chosen:
  ✅ SOTA Performance: Best NLI model on MNLI benchmark (89.4% accuracy)
  ✅ Robust: Handles complex reasoning and ambiguous cases
  ✅ Well-tested: Industry standard for zero-shot classification
  ✅ Facebook Research: High quality, actively maintained

How it works:
  Input: "Marie Curie: physicist and chemist"
  Hypothesis: "This entity is a material entity"
  → BART encodes premise + hypothesis
  → Predicts: Entailment (0.87), Neutral (0.10), Contradiction (0.03)

Trade-off:
  - Large (1.6GB), slower (~200ms/prediction)
  - Worth it for accuracy on ambiguous classifications

When to use alternatives:
  - DeBERTa-v3-mnli: Need absolute best (90.7%), have compute
  - distilbert-mnli: Speed critical, production latency requirements

Benchmarks:
  facebook/bart-large-mnli:    MNLI: 89.4%  Latency: 200ms  Size: 1.6GB
  DeBERTa-v3-mnli:            MNLI: 90.7%  Latency: 150ms  Size: 440MB
  distilbert-mnli:            MNLI: 82.2%  Latency: 60ms   Size: 260MB

3. FINE-TUNING BASE: distilbert-base-uncased
--------------------------------------------------------------------------------------
Why chosen:
  ✅ Training speed: 2x faster than BERT (60% fewer parameters)
  ✅ Size: 40% smaller (66M params vs 110M)
  ✅ Performance: 97% of BERT accuracy (GLUE: 79.9 vs 82.1)
  ✅ GPU friendly: Fits on smaller GPUs, faster iteration
  ✅ Good for demos: Quick experimentation

Trade-off:
  - Slight accuracy drop vs BERT
  - Worth it for 2x faster training and smaller memory footprint

When to use alternatives:
  - bert-base-uncased: Need max compatibility, more checkpoints available
  - deberta-v3-small: Want cutting-edge architecture (GLUE: 82.2)
  - roberta-base: Need robustness to noisy input

Benchmarks:
  distilbert-base:      GLUE: 79.9  Params: 66M   Size: 260MB
  bert-base:           GLUE: 82.1  Params: 110M  Size: 440MB
  deberta-v3-small:    GLUE: 82.2  Params: 44M   Size: 180MB
  roberta-base:        GLUE: 83.5  Params: 125M  Size: 500MB

======================================================================================
RECOMMENDED CONFIGURATIONS
======================================================================================

Production (Speed Priority):
  - Semantic: all-MiniLM-L6-v2 (50ms)
  - Zero-shot: distilbert-mnli (60ms)
  - Strategy: cascade (early exit)
  - Total latency: ~50-110ms

Research (Quality Priority):
  - Semantic: all-mpnet-base-v2 (200ms)
  - Zero-shot: DeBERTa-v3-mnli (150ms)
  - Strategy: ensemble (parallel)
  - Total latency: ~350ms

Resource-Constrained:
  - Semantic: all-MiniLM-L6-v2 (50ms)
  - Zero-shot: skip (save memory)
  - Strategy: hybrid_confidence
  - Total latency: ~80ms

======================================================================================
"""
    print(justifications)


if __name__ == "__main__":
    # Demo usage
    registry = ModelRegistry()

    print("Available presets:")
    for preset in registry.list_presets():
        config = registry.get_preset_config(preset)
        print(f"  - {preset}: {config['description']}")

    print("\n" + registry.compare_models('semantic'))
    print("\n" + registry.compare_models('zeroshot'))
    print("\n" + registry.compare_models('finetuned'))

    print_model_justifications()
