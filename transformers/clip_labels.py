from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LabelsClipper(BaseEstimator, TransformerMixin):
    """ Wraps the given regressor and returns its predictions rounded and clipped to 1-8. """

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y=None, **fit_params):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return np.round(np.clip(self.regressor.predict(X), 1, 8)).astype(int)