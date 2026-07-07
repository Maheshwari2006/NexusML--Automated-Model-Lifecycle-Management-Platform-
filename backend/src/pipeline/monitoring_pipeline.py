from backend.src.components.model_monitor import ModelMonitor
from backend.src.components.retraining import Retraining


class MonitoringPipeline:

    def run(self):

        monitor = ModelMonitor()

        monitor.monitor()

        import json

        with open(
            "backend/artifacts/monitoring_report.json"
        ) as file:

            report = json.load(file)

        if report["Status"] == "Retraining Required":

            retrain = Retraining()

            retrain.retrain()