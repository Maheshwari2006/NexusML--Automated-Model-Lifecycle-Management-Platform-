import json
import os


class ModelMonitor:

    def monitor(self):

        metrics_path = "backend/artifacts/metrics.json"

        with open(metrics_path, "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        report = {
            "Current Accuracy": accuracy,
            "Threshold": 0.80,
            "Status": ""
        }

        if accuracy < 0.80:
            report["Status"] = "Retraining Required"
        else:
            report["Status"] = "Model Healthy"

        os.makedirs(
            "backend/artifacts",
            exist_ok=True
        )

        with open(
            "backend/artifacts/monitoring_report.json",
            "w"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )
        print(report["Status"])
        print("Monitoring Report Saved")