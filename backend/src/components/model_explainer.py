import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


class ModelExplainer:

    def __init__(self):

        self.model = joblib.load(
            "backend/artifacts/model.pkl"
        )

    def explain(self, X):

        explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(X)

        os.makedirs("backend/artifacts", exist_ok=True)

        plt.figure()

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        plt.savefig(
            "backend/artifacts/shap_summary.png",
            bbox_inches="tight"
        )

        plt.close()

        return shap_values