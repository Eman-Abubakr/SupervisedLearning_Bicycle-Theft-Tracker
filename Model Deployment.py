# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:43:38 2022

@author: Eman Abubakr
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import pickle
import joblib
import sys
from os import path
from sklearn import metrics
from flask_cors import CORS

project_folder = r"C:/Users/Eman Abubakr/Downloads/Centennial/project"
models = {
         "Best_Model":"best_model_pipeline_All_Dataset.pkl"
         ,"Random_Forest": "Random forest_model.pkl"
         ,"K-nearest neighbors": "K-nearest neighbors_model.pkl"
         ,"Decision_Tree": "Decision Trees_model.pkl"
         ,"Logistic_Regression": "logistic regression_model.pkl"
         ,"Linear SVM": "linear svc_model.pkl"
         }

# data frames 

X_train_df = pd.read_csv(path.join(project_folder,"x_train.csv"))
y_train_df = pd.read_csv(path.join(project_folder,"y_train.csv"))
X_test_df = pd.read_csv(path.join(project_folder,"x_test.csv"))
y_test_df = pd.read_csv(path.join(project_folder,"y_test.csv"))

# Your API definition
app = Flask(__name__)
CORS(app)

@app.route("/predict/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def predict(model_name):
    if loaded_model:
        try:
            json_ = request.json
            print('JSON: \n', json_)
            query = pd.DataFrame(json_, columns=model_columns)
            prediction = list(loaded_model[model_name].predict(query))
            print(f'Returning prediction with {model_name} model:')
            print('prediction=', prediction)
            res = jsonify({"prediction": str(prediction)})
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
    
@app.route("/scores/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def scores(model_name):
    if loaded_model:
        try:
            y_pred = loaded_model[model_name].predict(X_test_df)
            print(f'Returning scores for {model_name}:')
            accuracy = metrics.accuracy_score(y_test_df, y_pred)
            precision = metrics.precision_score(y_test_df, y_pred)
            recall = metrics.recall_score(y_test_df, y_pred)
            f1 = metrics.f1_score(y_test_df, y_pred)
            print(f'accuracy={accuracy}  precision={precision}  recall={recall}  f1={f1}')
            res = jsonify({"accuracy": accuracy,
                            "precision": precision,
                            "recall":recall,
                            "f1": f1
                           })
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
        

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
        
    # load all models:
    loaded_model = {}
    for model_name in (models):
        loaded_model[model_name] = joblib.load(path.join(project_folder, models[model_name]))
        print(f'Model {model_name} loaded')
        
    model_columns = ['Primary_Offence',
           'Occurrence_DayOfWeek', 'Occurrence_DayOfYear', 'Occurrence_Hour',
           'Division','Premises_Type', 'Bike_Type',
           'Bike_Speed', 'Cost_of_Bike']
    
    
    app.run(port=port, debug=True)