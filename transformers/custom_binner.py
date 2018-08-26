import numbers

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class CustomBinner(BaseEstimator, TransformerMixin):
    """ Applies binning and adds bin number as a numerical variable. """

    # dict in form:
    #  { 'column_name': { 'bins': [int] | int, 'values': [int]}  }
    def __init__(self, configuration, nan=False, drop=False):
        self.nan = nan
        self.drop = drop
        self.configuration = configuration.copy()

    def fit(self, X, y=None, **fit_params):
        # if bins param is a number then use it as number of thresholds for equally distributed bins
        for column, config in self.configuration.items():
            if 'bins' in config and isinstance(config['bins'], numbers.Number):
                minimum = X[column].dropna().min()
                maximum = X[column].dropna().max()
                config['bins'] = np.linspace(minimum, maximum, config['bins'])
        return self

    def transform(self, X, y=None, **fit_params):
        for column, config in self.configuration.items():
            values = config['values'] if 'values' in config else []
            if self.nan:
                X["{}_{}".format(column , 'nan')] = X[column].isnull() * 1
            for val in values:
                X["{}_v{}".format(column, val)] = (X[column] == val) * 1
            if 'bins' in config:
                X["{}_bin".format(column)] = 0
                for i in range(len(config['bins']) - 1):
                    threshold = config['bins'][i]
                    X["{}_bin".format(column)] += (X[column] >= threshold) * 1
            if self.drop:
                X = X.drop(column, axis=1)
        return X

class CustomBinaryBinner(BaseEstimator, TransformerMixin):
    """ Applies binning and adds bin number as a one-hot variable. """

    # dict in form:
    #  { 'column_name': { 'bins': [int], 'values': [int]}  }
    def __init__(self, configuration, nan=False, drop=False):
        self.nan = nan
        self.drop = drop
        self.configuration = configuration.copy()

    def fit(self, X, y=None, **fit_params):
        for column, config in self.configuration.items():
            if 'bins' in config and isinstance(config['bins'], numbers.Number):
                minimum = X[column].dropna().min()
                maximum = X[column].dropna().max()
                config['bins'] = np.linspace(minimum, maximum, config['bins'])
        return self

    def transform(self, X, y=None, **fit_params):
        for column, config in self.configuration.items():
            values = config['values'] if 'values' in config else []
            if self.nan:
                X["{}_{}".format(column , 'nan')] = X[column].isnull() * 1
            for val in values:
                X["{}_v{}".format(column, val)] = (X[column] == val) * 1
            if 'bins' in config:
                for i in range(len(config['bins']) - 1):
                    lo = config['bins'][i]
                    hi = config['bins'][i+1]
                    X["{}_r{}_{}".format(column, lo, hi)] = ((X[column] >= lo) & (X[column] < hi)) * 1
            if self.drop:
                X = X.drop(column, axis=1)
        return X