
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MONITORING_DIR = BASE_DIR / "monitoring"

REFERENCE_DATA = ARTIFACTS_DIR / "processed_data.csv"
PREDICTIONS_FILE = MONITORING_DIR / "predictions.jsonl"
REPORT_FILE = ARTIFACTS_DIR / "monitoring_report.json"

FEATURES = ["Age", "MonthlyIncome", "YearsAtCompany"]


def log_prediction(age, income, years, prediction, model_version=None):
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "prediction": prediction,
        "model_version": model_version,
    }

    with PREDICTIONS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


def get_monitoring_summary():
    if not PREDICTIONS_FILE.exists():
        return {
            "total_predictions": 0,
            "message": "No predictions recorded yet.",
        }

    records = [
        json.loads(line)
        for line in PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return {
        "total_predictions": len(records),
        "prediction_counts": pd.Series(
            [record["prediction"] for record in records]
        ).value_counts().to_dict(),
        "latest_prediction": records[-1],
    }


def detect_data_drift():
    if not REFERENCE_DATA.exists():
        return {"status": "unavailable", "message": "Reference dataset not found."}

    if not PREDICTIONS_FILE.exists():
        return {"status": "insufficient_data", "message": "No prediction data yet."}

    reference = pd.read_csv(REFERENCE_DATA)
    records = [
        json.loads(line)
        for line in PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    current = pd.DataFrame(records)

    missing = [feature for feature in FEATURES
               if feature not in reference.columns or feature not in current.columns]
    if missing:
        return {"status": "error", "message": f"Missing features: {missing}"}

    results = {}
    for feature in FEATURES:
        reference_mean = float(reference[feature].dropna().mean())
        current_mean = float(current[feature].dropna().mean())

        results[feature] = {
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "mean_difference": current_mean - reference_mean,
        }

    return {
        "status": "completed",
        "method": "mean comparison; preliminary drift indicator",
        "features": results,
    }


def save_monitoring_report():
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": get_monitoring_summary(),
        "drift": detect_data_drift(),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
