from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


class ModelBasedColImputer(BaseEstimator, TransformerMixin):
    """
    Uses colname_nan to indicate which rows were nan
    """
    def __init__(self, column, model):
        self.column = column
        self.model = clone(model)

    def fit(self, X, y=None, **fit_params):
        without_na = X[~X[self.column + '_nan'].astype('bool')]
        without_target_col = without_na.drop([self.column, self.column + '_nan'], axis=1)
        self.model.fit(without_target_col, without_na[self.column])
        return self

    def transform(self, X):
        with_na = X[X[self.column + '_nan'].astype('bool')]
        without_target_col = with_na.drop([self.column, self.column + '_nan'], axis=1)
        X.loc[X[self.column + '_nan'].astype('bool'), self.column] = self.model.predict(without_target_col)
        return X

class ModelBasedFullImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, model):
        self.columns = columns
        self.model = model

    def fit(self, X, y=None, **fit_params):
        imputers = [(col + '_imputer', ModelBasedColImputer(column=col, model=self.model)) for col in self.columns if col + '_nan' in X.columns]
        self.pipe = Pipeline(imputers)
        return self.pipe.fit(X, y)

    def transform(self, X):
        return self.pipe.transform(X)
