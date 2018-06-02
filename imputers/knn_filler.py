from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from time import time

def get_nn_value(x, data, y, k):
    s = time()

    distances = np.linalg.norm(np.nan_to_num(data-x), axis=1)
    # distances = np.apply_along_axis(lambda x2: dist_fn(x2, x), 1, data)
    # distances = np.apply_along_axis(lambda x2: dist_fn(x2, x), 1, data)

    # print('d', time() - s)
    # s = time()
    ind = np.argpartition(distances, k)[:k]
    # print('s', time() - s)
    return np.mean(y[ind])

def dist_fn1(a, x):
    dist = np.nan_to_num(x - a)
    dist = np.sqrt(np.sum(np.power(dist, 2)))
    return dist



class KnnFiller(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k
        # self.dist_fn = dist_fn

    def fit(self, X, y, **fit_params):
        # self.ids = X.index.values
        self.data = X.values
        return self

    def transform(self, X):
        cols = X.columns[X.isna().any()].tolist()
        for col in cols:
            # s = time()
            j = X.columns.get_loc(col)
            rest = self.data[~np.isnan(self.data[:, j]), :]
            y = rest[:, j]
            rest = np.delete(rest, j, 1)
            # print('b', time() - s)
            for i, row in X.loc[X[col].isnull()].iterrows():
                if pd.isnull(row[col]):
                    X[col][i] = get_nn_value(row.drop(col).values, rest, y, self.k)

        return X