from django.shortcuts import render
import pandas as pd
import os
from django.views.decorators.csrf import csrf_exempt
from src.pipelines.prediction_pipeline import PredictionPipeline,CustomData

def is_float(x:str)->bool:
    try:
        float(x)
        return True
    except ValueError as e:
        return False

@csrf_exempt
def index(request):
    df=pd.read_csv(os.path.join("notebooks/data",'wafer.csv'))
    columns=list(df.columns)
    columns.pop()
    if request.method=='POST':
        dict1=request.POST
        values=[]
        for col in columns:
            if is_float(dict1[col]):
                values.append(float(dict1[col]))
            else:
                values.append(None)
        values.append(1)
        cst_data=CustomData(values)
        df=cst_data.get_data_frame()
        prediction=PredictionPipeline().predict(df)
        result=""
        if prediction[0]==2:
            result="Defective"
        else:
            result="Not Defective"
        return render(request,"index.html",{"list":columns,"result":result})
    return render(request,"index.html",{"list":columns})