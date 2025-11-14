"""
Classification Logger

Logs all classification attempts with full metadata to YAML files.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from models.wikidata import WikidataEntity
from models.results import ClassificationResult, ClassificationMatch


class ClassificationLogger:
    """Logs classification results to YAML files"""

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize logger

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_classification(
        self,
        entity: WikidataEntity,
        result: ClassificationResult,
        config: Dict[str, Any],
        device: str = "cpu",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a classification attempt

        Args:
            entity: The entity that was classified
            result: Classification result
            config: Configuration used
            device: Device used (cpu/cuda)
            additional_metadata: Any extra metadata to include

        Returns:
            Path to the log file
        """
        timestamp = datetime.now()
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'entity': {
                'id': entity.id,
                'label': entity.label,
                'description': entity.description,
                'text': entity.get_text(),
                'aliases': entity.aliases[:5] if entity.aliases else []
            },
            'classification': {
                'strategy': result.strategy,
                'processing_time_ms': result.processing_time_ms,
                'top_matches': [
                    {
                        'class_uri': match.class_uri,
                        'class_label': match.class_label,
                        'confidence': float(match.confidence),
                        'source': match.source,
                        'metadata': self._serialize_metadata(match.metadata)
                    }
                    for match in result.matches
                ]
            },
            'system_config': {
                'device': device,
                'models': {
                    'semantic': config.get('models', {}).get('semantic'),
                    'zeroshot': config.get('models', {}).get('zeroshot'),
                    'finetuned_base': config.get('models', {}).get('finetuned_base')
                },
                'thresholds': config.get('strategies', {}).get(result.strategy, {}).get('confidence_thresholds', {}),
                'classifier_config': {
                    'semantic': config.get('classifiers', {}).get('semantic', {}),
                    'zeroshot': config.get('classifiers', {}).get('zeroshot', {}),
                    'rule_based': config.get('classifiers', {}).get('rule_based', {})
                }
            }
        }

        # Add cascade decisions if available
        if result.matches and 'cascade_decisions' in result.matches[0].metadata:
            log_entry['cascade_trace'] = result.matches[0].metadata['cascade_decisions']

        # Add any additional metadata
        if additional_metadata:
            log_entry['additional_metadata'] = additional_metadata

        # Generate filename: YYYYMMDD_HHMMSS_EntityID.yaml
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{entity.id}.yaml"
        filepath = self.log_dir / filename

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(
                log_entry,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )

        return str(filepath)

    def log_batch_classification(
        self,
        classifications: List[Dict[str, Any]],
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log multiple classifications in a single batch file

        Args:
            classifications: List of classification log entries
            batch_metadata: Metadata about the batch run

        Returns:
            Path to the batch log file
        """
        timestamp = datetime.now()
        batch_entry = {
            'timestamp': timestamp.isoformat(),
            'batch_info': batch_metadata or {},
            'total_classifications': len(classifications),
            'classifications': classifications
        }

        # Generate filename: batch_YYYYMMDD_HHMMSS.yaml
        filename = f"batch_{timestamp.strftime('%Y%m%d_%H%M%S')}.yaml"
        filepath = self.log_dir / filename

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(
                batch_entry,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )

        return str(filepath)

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize metadata for YAML output

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Serialized metadata safe for YAML
        """
        if not metadata:
            return {}

        serialized = {}
        for key, value in metadata.items():
            # Convert numpy types to Python types
            if hasattr(value, 'item'):
                serialized[key] = value.item()
            # Convert lists of dicts (like cascade_decisions)
            elif isinstance(value, list):
                serialized[key] = [
                    {k: float(v) if isinstance(v, (int, float)) else v for k, v in item.items()}
                    if isinstance(item, dict) else item
                    for item in value
                ]
            # Keep other values as-is
            else:
                serialized[key] = value

        return serialized

    def get_recent_logs(self, n: int = 10) -> List[Path]:
        """
        Get the N most recent log files

        Args:
            n: Number of recent logs to retrieve

        Returns:
            List of log file paths
        """
        log_files = sorted(
            self.log_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return log_files[:n]

    def read_log(self, filepath: str) -> Dict[str, Any]:
        """
        Read a log file

        Args:
            filepath: Path to log file

        Returns:
            Parsed log entry
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
