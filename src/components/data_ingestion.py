import os 
import sys
from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join("artifacts",'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def __get_cols_with_zero_std_dev(self,df:pd.DataFrame)->list:
        logging.info("checking for the constant columns")
        cols_with_zero_std=list()
        # num_cols=[f for f in df.columns if df[f]!='object']
        for col in df.columns:
            if df[col].std()==0:
                cols_with_zero_std.append(col)
        logging.info("checking completed")
        return cols_with_zero_std
    
    def __get_redundant_cols(self,df:pd.DataFrame,missing_thresh=0.7):
        logging.info("checking for the redundant columns")
        cols_to_drop=list()
        cols_missing_ratio=df.isna().sum().div(df.shape[0])
        cols_to_drop=list(cols_missing_ratio[cols_missing_ratio>missing_thresh].index)
        logging.info("checking completed")
        return cols_to_drop
    
    def __drop_cols(self,df:pd.DataFrame):
        drop_cols1=self.__get_cols_with_zero_std_dev(df)
        drop_cols2=self.__get_redundant_cols(df)
        cols_to_drop=drop_cols1+drop_cols2
        # cols_to_drop.append("Unnamed: 0")
        # cols_to_drop.append("Good/Bad")
        df.drop(cols_to_drop,axis=1,inplace=True)
    def initiate_data_ingestion(self):
        logging.info("data ingestion initialized")

        try:
            logging.info("reading data set")
            df=pd.read_csv(os.path.join('notebooks/data','wafer.csv'))
            df.drop('Unnamed: 0',axis=1,inplace=True)
            self.__drop_cols(df)
            logging.info("data read successfully in pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)
            train_set,test_set=train_test_split(df,test_size=0.15,random_state=45)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("data ingestion successfully completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("error occured during data ingestion")