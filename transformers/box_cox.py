from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from tools.utils import box_cox


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lambdas_per_column = None):
        self.lambdas_per_column = lambdas_per_column

    def fit(self, X, y=None, **fit_params):
        # if self.columns is None:
        #     self.columns = X.select_dtypes(exclude = ["object"]).columns
        # skewness = X[self.columns].apply(lambda x: skew(x))
        # skewness = skewness[abs(skewness)>0.5]
        # self.skew_features = skewness.index
        return self

    def transform(self, X):
        for col, l in self.lambdas_per_column.items():
            X[col] = box_cox(X[col], l)
        return X