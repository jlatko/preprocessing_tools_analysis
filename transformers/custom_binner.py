from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CustomBinner(BaseEstimator, TransformerMixin):
    # dict in form:
    #  { 'column_name': { 'bins': [int], 'values': [int], 'nan': bool, 'drop': bool  }  }
    def __init__(self, configuration):
        self.configuration = configuration

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        for column, config in self.configuration.items():
            values = config['values'] if 'values' in config else []
            if 'nan'in config and config['nan']:
                X["{}_{}".format(column , 'nan')] = X[column].isnull() * 1
            for val in values:
                X["{}_v{}".format(column, val)] = (X[column] == val) * 1
            if 'bins' in config:
                for i in range(len(config['bins']) - 1):
                    lo = config['bins'][i]
                    hi = config['bins'][i+1]
                    X["{}_r{}_{}".format(column, lo, hi)] = ((X[column] >= lo) & (X[column] < hi)) * 1
            if 'drop' in config and config['drop']:
                X = X.drop(column, axis=1)
        return X