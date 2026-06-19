class PredictionPipeline:

    def predict(self, age, income, years):

        if income < 30000:
            return "Employee Likely To Leave"

        return "Employee Likely To Stay"
    
import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):
        self.model = joblib.load(
            "backend/artifacts/model.pkl"
        )

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