import os.path

from sensor.utils.main_utils import load_numpy_array_data,load_object
from sensor.exception import SensorException
import sys
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.entity.config_entity import ModelTrainingConfig

from sensor.ml.model.estimator import SensorModel

from sensor.utils.main_utils import save_object

from sensor.entity.artifact_entity import ModelTRainerArtifact

class ModelTrainer:
    def __init__(self,model_trainer_config,data_transformation_artifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            SensorException(e,sys)



    def trained_model(self,x_train,y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train,y_train)
        except Exception as e:
            raise SensorException(e,sys)


    def initiate_train_model(self):
        try:
            #address the train file path
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            #address the test_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path


            #loading train array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            #train the model
            model = self.trained_model(x_train,y_train)
            #prerdict the training data result
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")

            #predict the test data result
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            #overfitting and underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            #load the preprocess object(robust scaler and filling thre null values
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            #combining both the preprocessing model and the classification model

            sensor_model = SensorModel(preprocessor,model)
            save_object(self.model_trainer_config.trained_model_file_path,sensor_model)

            #model trainer artifact

            model_trainer_artifact = ModelTRainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          train_metric_artifact = classification_train_metric,
                                                          test_metric_artifact=classification_test_metric)
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)









