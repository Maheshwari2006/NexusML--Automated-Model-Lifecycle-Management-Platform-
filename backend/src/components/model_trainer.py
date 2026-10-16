
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "processed_data.csv"
ACTIVE_MODEL_PATH = ARTIFACTS_DIR / "model.pkl"

RETRAINING_DIR = BASE_DIR / "retraining"
CANDIDATE_MODEL_PATH = RETRAINING_DIR / "candidate_model.pkl"


class ModelTrainer:

    def train(self, save_as_candidate=False):
        """
        Train and evaluate a model.

        save_as_candidate=True:
            Save the model separately without replacing the active model.

        save_as_candidate=False:
            Preserve the original training behavior and save the active model.
        """

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {DATA_PATH}"
            )

        df = pd.read_csv(DATA_PATH)

        features = [
            "Age",
            "MonthlyIncome",
            "YearsAtCompany",
        ]
        target = "Attrition"

        missing = [
            col for col in features + [target]
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required dataset columns: {missing}"
            )

        df = df[features + [target]].dropna()

        X = df[features]
        y = df[target]

        if y.nunique() < 2:
            raise ValueError(
                "Training requires at least two target classes."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        with mlflow.start_run():
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = {
                "accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "f1_score": float(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("save_as_candidate", save_as_candidate)
            mlflow.log_metrics(metrics)

            if os.getenv("GITHUB_ACTIONS") != "true":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="EmployeeAttritionModel",
                )
                print("Model logged to MLflow.")
            else:
                print(
                    "GitHub Actions detected - "
                    "skipping MLflow model logging."
                )

        if save_as_candidate:
            RETRAINING_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            output_path = CANDIDATE_MODEL_PATH
        else:
            ARTIFACTS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            output_path = ACTIVE_MODEL_PATH

        joblib.dump(model, output_path)

        print(f"Model saved to: {output_path}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

        return {
            "model_path": str(output_path),
            "metrics": metrics,
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
        }
