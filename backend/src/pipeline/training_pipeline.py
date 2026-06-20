from backend.src.components.data_ingestion import DataIngestion
from backend.src.components.data_validation import DataValidation
from backend.src.components.data_transformation import DataTransformation
from backend.src.components.model_trainer import ModelTrainer
from backend.src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:

    def start_training_pipeline(self):

        print("Starting Training Pipeline")

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

        print("Training Pipeline Completed")