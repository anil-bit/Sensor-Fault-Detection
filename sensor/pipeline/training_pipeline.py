from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainingConfig,ModelEvaluationConfig,ModelPusherConfig
from sensor .exception import SensorException
import sys,os

from sensor.logger import logging
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.artifact_entity import DataValidationArtifact
from sensor.entity.artifact_entity import DataTransformationArtifact
from sensor.entity.artifact_entity import ModelTrainerArtifact
from sensor.entity.artifact_entity import ModelEvaluationArtifact

from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher

from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME,PREDICTION_BUCKET_NAME
from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.constant.training_pipeline import SAVED_MODEL_DIR



class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        training_pipeline_config = TrainingPipelineConfig()
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        self.data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        self.training_pipeline_config = training_pipeline_config
        self.data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        self.model_training_config = ModelTrainingConfig(training_pipeline_config)
        self.model_evaluation_config = ModelEvaluationConfig(training_pipeline_config)
        self.model_pusher_config = ModelPusherConfig(training_pipeline_config)
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            #self.data_ingestion_config = DataIngestionConfig
            logging.info("starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e,sys)


    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e,sys)


    def start_data_transformation(self,data_validation_artifact):
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            #initialize the data transformation class
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=self.data_transformation_config)
            #implimenting the data transformation
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)



    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        logging.info("start model trainer method")
        try:
            #initialize the model training
            model_trainer = ModelTrainer(self.model_training_config,data_transformation_artifact)
            model_trainer_artifact =  model_trainer.initiate_train_model()
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)

    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact):
        try:
            #initialize the model evaluation
            model_evaluation = ModelEvaluation(self.model_evaluation_config,data_validation_artifact, model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise SensorException(e,sys)


    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(self.model_pusher_config,model_evaluation_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e,sys)



    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys)

    def sync_saved_model_to_s3(self):
        try:
            aws_bucket_url = f"s3://{PREDICTION_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder=SAVED_MODEL_DIR,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys)




    def run_pipeline(self):
        try:
            data_ingestion_artifact:DataIngestionArtifact=self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_validation_artifact,model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            TrainPipeline.is_pipeline_running=False
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_to_s3()

        except Exception as e:
            self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise SensorException(e,sys)


