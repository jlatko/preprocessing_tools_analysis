from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


class HotDeckSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, k=8):
        self.columns = columns
        self.k = k

    def fit(self, X, y=None, **fit_params):
        self.clusterer = KMeans(n_clusters=self.k)
        without_na = X.dropna(axis=0)

        # fit KMeans to other columns
        without_target_cols = without_na.drop(self.columns, axis=1)
        self.clusterer.fit(without_target_cols)

        just_target_cols = without_na[self.columns]
        # get mean of specified attributes per each cluster
        just_target_cols['cluster'] = self.clusterer.predict(without_target_cols)
        self.values_per_cluster = just_target_cols.groupby('cluster').apply(np.mean)
        return self

    def transform(self, X):
        rows_with_nan = X[X[self.columns].isnull().any(axis=1)]
        without_target_cols = rows_with_nan.drop(self.columns, axis=1)
        rows_with_nan['cluster'] = self.clusterer.predict(without_target_cols)
        for col in self.columns:
            X.loc[X[col].isnull(), col] = rows_with_nan['cluster'].map(self.values_per_cluster[col])
        return X
