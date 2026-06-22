import pandas as pd

class DriftDetector:

    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect_drift(self, train_df, current_df):

        train_mean = train_df.mean(numeric_only=True)
        current_mean = current_df.mean(numeric_only=True)

        drift_score = abs(train_mean - current_mean).mean()

        print(f"Drift Score: {drift_score:.4f}")

        if drift_score > self.threshold:
            return True

        return False