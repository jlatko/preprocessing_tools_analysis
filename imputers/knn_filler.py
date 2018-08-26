from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def get_nn_value(x, data, y, k):
    distances = np.linalg.norm(np.nan_to_num(data-x), axis=1)
    ind = np.argpartition(distances, k)[:k]
    return np.mean(y[ind])


def dist_fn1(a, x):
    dist = np.nan_to_num(x - a)
    dist = np.sqrt(np.sum(np.power(dist, 2)))
    return dist


class KnnFiller(BaseEstimator, TransformerMixin):
    """ Imputes blanks using values from nearest neighbours. """

    def __init__(self, k):
        self.k = k

    def fit(self, X, y, **fit_params):
        self.data = X.values
        return self

    def transform(self, X):
        cols = X.columns[X.isna().any()].tolist()
        for col in cols:
            j = X.columns.get_loc(col)
            rest = self.data[~np.isnan(self.data[:, j]), :]
            y = rest[:, j]
            rest = np.delete(rest, j, 1)
            for i, row in X.loc[X[col].isnull()].iterrows():
                if pd.isnull(row[col]):
                    X[col][i] = get_nn_value(row.drop(col).values, rest, y, self.k)
        return X