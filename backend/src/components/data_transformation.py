import pandas as pd


class DataTransformation:

    def transform(self):

        df = pd.read_csv(
            "backend/data/raw/employee.csv"
        )

        processed_df = df[
            [
                "Age",
                "MonthlyIncome",
                "YearsAtCompany",
                "Attrition"
            ]
        ].copy()

        processed_df["Attrition"] = (
            processed_df["Attrition"]
            .map({
                "Yes": 1,
                "No": 0
            })
        )

        processed_df.to_csv(
            "backend/artifacts/processed_data.csv",
            index=False
        )

        print(
            "Data Transformation Completed"
        )