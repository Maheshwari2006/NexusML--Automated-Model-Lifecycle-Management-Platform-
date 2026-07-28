import pandas as pd

from backend.src.components.model_explainer import ModelExplainer


class ExplanationPipeline:

    def run_pipeline(self):

        data = pd.read_csv(
            "backend/artifacts/processed_data.csv"
        )

        X = data[
            [
                "Age",
                "MonthlyIncome",
                "YearsAtCompany"
            ]
        ]

        explainer = ModelExplainer()

        explainer.explain(X)

        print("SHAP Explanation Generated Successfully")