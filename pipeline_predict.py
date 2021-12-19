import joblib
import pandas as pd
import numpy as np

#Cargamos modelo y pipeline
kidney_disease_model = joblib.load('kidney_disease_pipeline.pkl')

#Funcion para hacer predicciones.
def predict(X):
    predicts = kidney_disease_model.predict(X)
    target_mapping = {0:'notckd', 1:'ckd'}
    predicts = pd.Series(predicts).map(target_mapping)
    return predicts[0]