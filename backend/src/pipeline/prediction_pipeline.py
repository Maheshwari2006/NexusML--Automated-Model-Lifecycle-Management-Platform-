from pathlib import Path
import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "model.pkl"
        )

        self.model = joblib.load(model_path)

    def predict(self, age, income, years):

        data = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [income],
            "YearsAtCompany": [years]
        })

        prediction = self.model.predict(data)

        if prediction[0] == 1:
            return "Employee Likely To Leave"

        return "Employee Likely To Stay"