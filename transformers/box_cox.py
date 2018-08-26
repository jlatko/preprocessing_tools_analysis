from sklearn.base import BaseEstimator, TransformerMixin

from tools.utils import box_cox

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """ Applies Box-Cox transformation according to the given config. """

    def __init__(self, lambdas_per_column = None):
        self.lambdas_per_column = lambdas_per_column

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        for col, l in self.lambdas_per_column.items():
            X[col] = box_cox(X[col], l)
        return X