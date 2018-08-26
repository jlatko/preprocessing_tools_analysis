from sklearn.base import BaseEstimator, TransformerMixin

class FeatureDropper(BaseEstimator, TransformerMixin):
    """ Drops specified columns. """

    def __init__(self, drop=[]):
        self.drop = drop

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[list(set(X.columns) - set(self.drop))]