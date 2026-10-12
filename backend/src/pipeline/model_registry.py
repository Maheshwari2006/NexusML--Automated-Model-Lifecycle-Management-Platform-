
from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

import joblib


BASE_DIR = Path(__file__).resolve().parents[2]
REGISTRY_DIR = BASE_DIR / "model_registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ACTIVE_MODEL = ARTIFACTS_DIR / "model.pkl"


def _load_registry():
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {"active_version": None, "versions": []}


def register_model(metrics=None):
    """Save the current model and metrics as a new version."""
    if not ACTIVE_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {ACTIVE_MODEL}")

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}

    registry = _load_registry()
    version = len(registry["versions"]) + 1
    version_dir = REGISTRY_DIR / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=False)

    version_model = version_dir / "model.pkl"
    shutil.copy2(ACTIVE_MODEL, version_model)

    version_metrics = version_dir / "metrics.json"
    version_metrics.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    registry["versions"].append({
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(version_model.relative_to(BASE_DIR)),
        "metrics": metrics,
    })
    registry["active_version"] = version

    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    return registry


def rollback_model(version: int):
    """Activate a previously registered model version."""
    registry = _load_registry()
    entry = next(
        (item for item in registry["versions"]
         if item["version"] == version),
        None,
    )
    if entry is None:
        raise ValueError(f"Model version v{version} does not exist.")

    source = BASE_DIR / entry["model_path"]
    if not source.exists():
        raise FileNotFoundError(f"Versioned model not found: {source}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, ACTIVE_MODEL)

    registry["active_version"] = version
    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    return registry


def list_versions():
    return _load_registry()
