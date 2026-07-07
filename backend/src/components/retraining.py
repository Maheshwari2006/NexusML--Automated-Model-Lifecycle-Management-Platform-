class Retraining:

    def retrain(self):

        print("Starting Retraining Pipeline...")

        from backend.src.components.model_trainer import ModelTrainer

        trainer = ModelTrainer()

        trainer.train()

        print("Retraining Completed")