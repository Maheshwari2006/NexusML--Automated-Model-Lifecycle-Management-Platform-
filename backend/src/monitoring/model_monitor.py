import json

class ModelMonitor:

    def compare_models(self, old_acc, new_acc):

        print(f"Old Accuracy: {old_acc}")
        print(f"New Accuracy: {new_acc}")

        return new_acc > old_acc

    def save_report(self, metrics):

        with open("backend/artifacts/monitoring_report.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print("Monitoring Report Saved")