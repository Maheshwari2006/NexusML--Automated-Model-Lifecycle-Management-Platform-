import pandas as pd
import os


class DataIngestion:

    def __init__(self):

        self.output_path = "backend/data/raw/employee.csv"

    def initiate_data_ingestion(self):

        os.makedirs(
            "backend/data/raw",
            exist_ok=True
        )

        url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"

        df = pd.read_csv(url)

        df.to_csv(
            self.output_path,
            index=False
        )

        print("Data Ingestion Completed")

        return self.output_path