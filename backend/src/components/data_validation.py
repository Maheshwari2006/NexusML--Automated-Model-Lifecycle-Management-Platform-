import pandas as pd
import yaml
import os


class DataValidation:

    def __init__(self):

        self.data_path = "backend/data/raw/employee.csv"
        self.schema_path = "backend/config/schema.yaml"

    def validate_dataset(self):

        df = pd.read_csv(self.data_path)

        with open(self.schema_path, "r") as file:
            schema = yaml.safe_load(file)

        required_columns = schema["columns"]

        validation_status = True

        for column in required_columns:

            if column not in df.columns:
                validation_status = False

        os.makedirs(
            "backend/artifacts",
            exist_ok=True
        )

        with open(
            "backend/artifacts/validation_status.txt",
            "w"
        ) as file:

            file.write(
                f"Validation Status: {validation_status}"
            )

        print(
            f"Validation Status: {validation_status}"
        )

        return validation_status