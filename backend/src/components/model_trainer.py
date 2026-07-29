import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def train(self):

        # Load processed dataset
        df = pd.read_csv("backend/artifacts/processed_data.csv")

        # Use only the features collected by the web form
        X = df[[
            "Age",
            "MonthlyIncome",
            "YearsAtCompany"
        ]]

        # Target column
        y = df["Attrition"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        with mlflow.start_run():

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            print(f"Accuracy : {accuracy:.4f}")

            # Log parameters
            mlflow.log_param("n_estimators", 100)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Log model (skip registry in GitHub Actions)
            if os.getenv("GITHUB_ACTIONS") != "true":

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="EmployeeAttritionModel"
                )

                print("Model Registered Successfully")

            else:

                print("GitHub Actions detected - Skipping MLflow Model Registry")

            # Save trained model
            os.makedirs("backend/artifacts", exist_ok=True)

            joblib.dump(
                model,
                "backend/artifacts/model.pkl"
            )

            print("Model Saved Successfully")