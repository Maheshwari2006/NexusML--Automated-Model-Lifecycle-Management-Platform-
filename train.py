from backend.src.components.data_ingestion import DataIngestion
from backend.src.components.data_validation import DataValidation
from backend.src.components.data_transformation import DataTransformation
from backend.src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    print("Starting MLOps Pipeline")

    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()

    validation = DataValidation()

    if validation.validate_dataset():

        transformation = DataTransformation()
        transformation.transform()

        trainer = ModelTrainer()
        trainer.train()

    print("Pipeline Completed")