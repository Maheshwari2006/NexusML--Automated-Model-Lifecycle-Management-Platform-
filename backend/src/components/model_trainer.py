import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def train(self):

        df = pd.read_csv(
            "backend/artifacts/processed_data.csv"
        )

        X = df.drop(
            columns=["Attrition"]
        )

        y = df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Accuracy : {accuracy:.2f}"
        )

        os.makedirs(
            "backend/artifacts",
            exist_ok=True
        )

        joblib.dump(
            model,
            "backend/artifacts/model.pkl"
        )

        print(
            "Model Saved Successfully"
        )