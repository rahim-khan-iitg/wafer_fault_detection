import os
import sys
import pickle
from src.logger import logging
from src.exceptions import CustomException
from sklearn.metrics import accuracy_score

def save_objects(file_path:str,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        pickle.dump(obj,open(file_path,'wb'))
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models:dict):
    logging.info("model evaluation initiated")
    try:
        report=dict()
        best_model=None
        best_acc=-1
        for name,model in models.items():
            m=model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            acc=accuracy_score(y_test,y_pred)
            if acc>best_acc:
                best_acc=acc
                best_model=m
            report[name]=acc
        logging.info("model evaluation completed")
        logging.info(f"best model is {best_model} with accuracy {best_acc}")
        return (report,m)
    except Exception as e:
        logging.info("error occured during model evaluation")
        raise CustomException(e,sys)

def load_objects(file_path):
    obj=pickle.load(open(file_path,'rb'))
    return obj