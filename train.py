from backend.src.components.data_ingestion import DataIngestion
from backend.src.components.data_validation import DataValidation
from backend.src.components.data_transformation import DataTransformation
from backend.src.components.model_trainer import ModelTrainer
from backend.src.components.model_evaluation import ModelEvaluation


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

        evaluator = ModelEvaluation()
        evaluator.evaluate()

    print("Pipeline Completed")