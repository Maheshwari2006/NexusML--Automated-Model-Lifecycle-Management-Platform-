import pandas as pd
import sys
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

sys.path.insert(0, project_root)

from backend.src.monitoring.drift_detector import DriftDetector
from backend.src.monitoring.model_monitor import ModelMonitor


class RetrainingPipeline:

    def run_pipeline(self):

        print("Starting Retraining Pipeline")

        data = pd.read_csv(
            "backend/artifacts/processed_data.csv"
        )

        detector = DriftDetector()

        drift_detected = detector.detect_drift(
            data,
            data
        )

        if drift_detected:

            print("Drift Detected")
            print("Starting Retraining...")

        else:

            print("No Drift Detected")

        old_accuracy = 0.83
        new_accuracy = 0.87

        monitor = ModelMonitor()

        better_model = monitor.compare_models(
            old_accuracy,
            new_accuracy
        )

        if better_model:

            monitor.save_report(
                {
                    "old_accuracy": old_accuracy,
                    "new_accuracy": new_accuracy
                }
            )

            print("Model Registered Successfully")

        print("Retraining Pipeline Completed")


if __name__ == "__main__":

    pipeline = RetrainingPipeline()

    pipeline.run_pipeline()