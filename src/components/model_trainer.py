import os
import sys
from src.exceptions import CustomException
from src.logger import logging

from dataclasses import dataclass

from src.utils import save_objects
from src.utils import evaluate_models

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
@dataclass
class ModelTrainerConfig:
    model_trainer_config=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,x_train,y_train,x_test,y_test):
        logging.info("model training initiated")
        try:
            models={
                "logistic":LogisticRegression(),
                "SVC":SVC(),
                "Decision Tree":DecisionTreeClassifier(),
                "adaboost":AdaBoostClassifier(),
                "Gradient":GradientBoostingClassifier(),
                "Random forest":RandomForestClassifier(),
            }
            report,best_model=evaluate_models(x_train,y_train,x_test,y_test,models)
            logging.info(f"training report:\n{report}")
            logging.info("model training completed")
            logging.info("saving the bast model")
            save_objects(
                file_path=self.trainer_config.model_trainer_config,
                obj=best_model
            )
        except Exception as e:
            logging.info("error occured during model training")
            raise CustomException(e,sys)
        
