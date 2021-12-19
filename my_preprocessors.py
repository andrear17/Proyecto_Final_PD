import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class Mapper(BaseEstimator, TransformerMixin):
    # Constructor
    # mappings --> mapeo de las variables categóricas
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError('Las variables deben ser incluidas en una lista.')
        
        # Campos de la clase Mapper
        self.variables = variables
        self.mappings = mappings

    # Método fit no hace nada pero es necesario para el pipeline
    def fit(self, X, y=None):
        return self

    # Método transform
    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].map(self.mappings)
        return X