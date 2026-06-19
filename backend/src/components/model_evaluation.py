import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class ModelEvaluation:

    def evaluate(self):

        os.makedirs(
            "backend/artifacts",
            exist_ok=True
        )

        df = pd.read_csv(
            "backend/artifacts/processed_data.csv"
        )

        X = df.drop(columns=["Attrition"])
        y = df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = joblib.load(
            "backend/artifacts/model.pkl"
        )

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(round(accuracy_score(y_test, predictions), 4)),
            "precision": float(round(precision_score(y_test, predictions), 4)),
            "recall": float(round(recall_score(y_test, predictions), 4)),
            "f1_score": float(round(f1_score(y_test, predictions), 4))
        }

        with open(
            "backend/artifacts/metrics.json",
            "w"
        ) as f:
            json.dump(metrics, f, indent=4)

        cm = confusion_matrix(
            y_test,
            predictions
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.savefig(
            "backend/artifacts/confusion_matrix.png"
        )

        plt.close()

        print("Metrics File Saved")
        print("Confusion Matrix Saved")
        print(metrics)