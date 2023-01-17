import pandas as pd
from sensor.entity.artifact_entity import DataValidationArtifact
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.exception import SensorException
from sensor.entity.config_entity import DataValidationConfig
import os,sys
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.logger import logging
from scipy.stats import ks_2samp
from sensor.utils.main_utils import write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys)



    def validate_number_of_columns(self,dataframe) -> bool:
        try:
            no_of_columns = len(self._schema_config["columns"])
            logging.info(f"required number of columns: {no_of_columns}")
            logging.info(f"data frame has columns: {len(dataframe.columns)}")
            print("no_of_columns wanttttttttttttttttttt")
            if len(dataframe.columns) == no_of_columns:
                return True
            return False
        except Exception as e:
            SensorException(e,sys)




    def is_numerical_column_exist(self,df) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = df.columns
            missing_numerical_columns = []
            numerical_columns_present = True
            for columns in numerical_columns:
                if columns not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(columns)
            logging.info(f"missing numerical columns:{missing_numerical_columns}")
            return numerical_columns_present
        except Exception as e:
            raise SensorException(e,sys)






    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e,sys)

    def detect_dataset_drift(self,base_df,current_df,threshold = 0.5):
        try:
            status = True
            report={}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                p_value = ks_2samp(d1,d2)
                if threshold<=p_value.pvalue:
                    is_found  = False
                else:
                    is_found = True
                    status = False
                report.update({col:{
                    "p_value":float(p_value.pvalue),
                    "drift_status":is_found}})

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #create directory

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(drift_report_file_path,content=report)


            return status
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            #reading data fgrom train and test file location
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            #validated number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all columns"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all columns"

            #validate numerical columns

            status = self.is_numerical_column_exist(df=train_dataframe)
            if not status:
                error_message = f"{error_message}train data does not contain all numerical values"
            status = self.is_numerical_column_exist(df = test_dataframe)
            if not status:
                error_message = f"{error_message}test data doe not contain all numerical values"

            if len(error_message)>0:
                raise Exception(error_message)
            #lets check the data drift
            status = self.detect_dataset_drift(train_dataframe,test_dataframe)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise SensorException(e,sys)
