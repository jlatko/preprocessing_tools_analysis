from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


class HotDeckColImputer(BaseEstimator, TransformerMixin):
    """
    Uses colname_nan to indicate which rows were nan
    """
    def __init__(self, column, k=8):
        self.column = column
        self.k = k

    def fit(self, X, y=None, **fit_params):
        self.clusterer = KMeans(n_clusters=self.k)

        without_na = X[~X[self.column + '_nan'].astype('bool')]
        with_na = X[X[self.column + '_nan'].astype('bool')]

        # fit KMeans to other columns
        without_target_col = without_na.drop([self.column, self.column + '_nan'], axis=1)
        # print(without_target_col.shape[0])
        self.clusterer.fit(without_target_col)

        just_target_col = without_na[[self.column]]
        # get mean of specified attributes per each cluster
        just_target_col['cluster'] = self.clusterer.predict(without_target_col)
        self.values_per_cluster = just_target_col.groupby('cluster').apply(np.mean)[self.column]
        return self

    def transform(self, X):
        with_na = X[X[self.column + '_nan'].astype('bool')]
        without_target_col = with_na.drop([self.column, self.column + '_nan'], axis=1)
        if without_target_col.shape[0]:
            with_na['cluster'] = self.clusterer.predict(without_target_col)
            X.loc[X[self.column + '_nan'].astype('bool'), self.column] = with_na['cluster'].map(self.values_per_cluster)
        return X

class HotDeckFullImputer(BaseEstimator, TransformerMixin):
    def __init__(self, col_k_pairs, default_k=8):
        """
        col_k_pairs: list of tuples (column name, k or none)
        """
        self.col_k_pairs = col_k_pairs
        self.default_k = default_k

    def fit(self, X, y=None, **fit_params):
        imputers = [
            (col + '_imputer', HotDeckColImputer(column=col, k=(k or self.default_k)))
            for col, k in self.col_k_pairs
            if col + '_nan' in X.columns and ( 0 < X[col + '_nan'].sum() < X.shape[0]) ]
        self.pipe = Pipeline(imputers)
        return self.pipe.fit(X, y)

    def transform(self, X):
        return self.pipe.transform(X)
