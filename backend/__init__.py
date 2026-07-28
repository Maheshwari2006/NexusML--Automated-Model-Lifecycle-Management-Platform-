from pathlib import Path
import joblib

class PredictionPipeline:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parents[3]
            / "backend"
            / "artifacts"
            / "model.pkl"
        )

        self.model = joblib.load(model_path)