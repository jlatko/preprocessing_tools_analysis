from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    """ Selects only specified features and drops the rest. """

    def __init__(self, features=[]):
        self.features = features

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[list(set(X.columns) & set(self.features))]