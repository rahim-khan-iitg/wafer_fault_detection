import os
import sys
from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from src.utils import save_objects

@dataclass
class DataTransformationConfig:
    data_transformation_config_path=os.path.join('artifacts','perprocessor.pkl')
    used_features_path=os.path.join("artifacts","features.pkl")


class DataTransformation:
    def __init__(self):
        self.data_tansformation_config=DataTransformationConfig()
    
    def __resample_data(self,df:pd.DataFrame):
        majority_class=df[df['Good/Bad']==0]
        minority_class=df[df["Good/Bad"]==2]
        if minority_class.shape[0]>majority_class.shape[0]:
            majority_class,minority_class=minority_class,majority_class

        minority_class_resampled=resample(minority_class,replace=True,n_samples=len(majority_class),random_state=45)
        df_resampled=pd.concat([majority_class,minority_class_resampled])
        return df_resampled

    
    def get_data_transformation_obj(self):
        pre_processing_pipeline=Pipeline(
                steps=[
                    ("imputer",KNNImputer(n_neighbors=3)),
                    ("scaler",RobustScaler())
                ]
        )
        return pre_processing_pipeline

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("reading training and testing data")
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            logging.info("Reading data completed")
            logging.info(f"Train DataFrame head : \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame head : \n{test_data.head().to_string()}")
            logging.info("obtaining the preprocessing object")
            pre_processing_obj=self.get_data_transformation_obj()

            target_column="Good/Bad"
            train_target=train_data[target_column]
            test_target=test_data[target_column]
            
            train_data=pre_processing_obj.fit_transform(train_data)
            test_data=pre_processing_obj.transform(test_data)
            logging.info("Preprocessing completed")
            logging.info("resampling is initiated")
            df1=pd.DataFrame(data=train_data,columns=pre_processing_obj.get_feature_names_out())
            resampled_data=self.__resample_data(df1)
            logging.info("resempling done")
            train_target=resampled_data['Good/Bad']
            resampled_data.drop('Good/Bad',axis=1,inplace=True)
            df2=pd.DataFrame(data=test_data,columns=pre_processing_obj.get_feature_names_out())
            test_target=df2['Good/Bad']
            df2.drop("Good/Bad",axis=1,inplace=True)
            logging.info("Resampling completed")
            logging.info("saving the preprocessor")
            save_objects(file_path=self.data_tansformation_config.data_transformation_config_path,obj=pre_processing_obj)
            logging.info("saving the used features")
            save_objects(file_path=self.data_tansformation_config.used_features_path,obj=pre_processing_obj.get_feature_names_out())
            return(
                resampled_data,
                train_target,
                df2,
                test_target
            )

        except Exception as e:
            logging.info("exception during data transformation")
            raise CustomException(e,sys)