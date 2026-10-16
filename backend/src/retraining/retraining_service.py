
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from backend.src.components.model_trainer import ModelTrainer
from backend.src.pipeline.model_registry import register_model


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RETRAINING_DIR = BASE_DIR / "retraining"

DATA_PATH = ARTIFACTS_DIR / "processed_data.csv"
ACTIVE_MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
CANDIDATE_MODEL_PATH = RETRAINING_DIR / "candidate_model.pkl"
REPORT_PATH = RETRAINING_DIR / "retraining_report.json"

FEATURES = ["Age", "MonthlyIncome", "YearsAtCompany"]
TARGET = "Attrition"


def retrain_and_evaluate():
    """Train a candidate and promote it only if evaluation passes."""

    if not DATA_PATH.exists():
        return {
            "status": "error",
            "message": "Training dataset not found.",
        }

    if not ACTIVE_MODEL_PATH.exists():
        return {
            "status": "error",
            "message": "Active model not found.",
        }

    df = pd.read_csv(DATA_PATH)

    missing = [
        column for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        return {
            "status": "error",
            "message": "Required dataset columns are missing.",
            "missing_columns": missing,
            "available_columns": df.columns.tolist(),
        }

    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        return {
            "status": "error",
            "message": "Dataset must contain at least two target classes.",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    active_model = joblib.load(ACTIVE_MODEL_PATH)
    active_predictions = active_model.predict(X_test)
    active_accuracy = float(
        accuracy_score(y_test, active_predictions)
    )

    # Train a candidate without replacing the active model.
    trainer = ModelTrainer()
    trainer_result = trainer.train(save_as_candidate=True)

    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    candidate_predictions = candidate_model.predict(X_test)
    candidate_accuracy = float(
        accuracy_score(y_test, candidate_predictions)
    )

    promoted = candidate_accuracy > active_accuracy

    if promoted:
        backup_path = RETRAINING_DIR / "previous_active_model.pkl"
        joblib.copy if False else None
        import shutil
        shutil.copy2(ACTIVE_MODEL_PATH, backup_path)

        joblib.dump(candidate_model, ACTIVE_MODEL_PATH)

       
        try:
            register_model(
                metrics={
                    "accuracy": candidate_accuracy,
                    "previous_accuracy": active_accuracy,
                    "promotion_reason": "Candidate improved accuracy",
                }
            )
        except Exception:
            shutil.copy2(backup_path, ACTIVE_MODEL_PATH)
            raise

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promoted" if promoted else "not_promoted",
        "active_accuracy_before": active_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "improvement": candidate_accuracy - active_accuracy,
        "candidate_metrics": trainer_result["metrics"],
        "active_model_replaced": promoted,
        "promotion_rule": "Candidate accuracy must exceed active accuracy",
    }

    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report
