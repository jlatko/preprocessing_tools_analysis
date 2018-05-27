from sklearn.base import BaseEstimator, TransformerMixin

class AutomaticDropper(BaseEstimator, TransformerMixin):
    def __init__(self, treshold):
        pass

    def fit(self, X, y=None, **fit_params):
        self.drop = []
        return self

    def transform(self, X, y=None, **fit_params):
        return X[list(set(X.columns) - set(self.drop))]