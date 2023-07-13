import os
import sys
from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    print("data ingestion completed")
    obj2=DataTransformation()
    x_train,y_train,x_test,y_test=obj2.initiate_data_transformation(train_data_path,test_data_path)
    print("x train shape",x_train.shape)
    print("x test shape",x_test.shape)
    print("y train shape",y_train.shape)
    print("y test shape",y_test.shape)
    print("transformation completed")
    obj3=ModelTrainer()
    obj3.initiate_model_training(x_train,y_train,x_test,y_test)
    print("model training completed")
