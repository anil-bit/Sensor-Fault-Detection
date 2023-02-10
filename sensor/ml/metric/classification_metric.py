from sensor.exception import SensorException
import sys
from sklearn.metrics import f1_score,recall_score,precision_score

from sensor.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(y_true,y_pred):
    try:
        #calculate f1 score
        model_f1_score = f1_score(y_true=y_true,y_pred=y_pred)
        #calculate recall score
        model_recall_score = recall_score(y_true,y_pred)
        #calculate precision score
        model_precision_score =  precision_score(y_true,y_pred)
        #save the final result in artifact
        classification_metric = ClassificationMetricArtifact(f1_score=model_f1_score,
                                                             precision_score=model_precision_score,
                                                             recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise SensorException(e,sys)
