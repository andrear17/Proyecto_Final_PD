import joblib
import pandas as pd
import numpy as np

#Cargamos modelo y pipeline
kidney_disease_model = joblib.load('kidney_disease_pipeline.pkl')
X_test = pd.read_csv('X_test.csv')

#Funcion para hacer predicciones.
def predict(X):
    predicts = kidney_disease_model.predict(X)
    return predicts[0]

#print(predict(X_test))
#X_test = pd.json_normalize({"PassengerId":pd.NA,"Pclass":1,"Name": "Mike Dallas","Sex": "male","Age": 22,"SibSp": 0,"Parch": 0,"Ticket": 649,"Fare": 7.25,"Cabin": "B14","Embarked":"S"})
#print('Resultado', predict(X_test))