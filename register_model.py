
import json
from pathlib import Path

from backend.src.pipeline.model_registry import register_model

metrics_path = Path("backend/artifacts/metrics.json")

metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

registry = register_model(metrics)
print(json.dumps(registry, indent=2))
