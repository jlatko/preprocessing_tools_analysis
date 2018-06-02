from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LabelsClipper(BaseEstimator, TransformerMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y=None, **fit_params):
        self.regressor.fit(X, y)
        print(X.shape)
        return self

    def predict(self, X):
        print(X.shape)
        return np.round(np.clip(self.regressor.predict(X), 1, 8)).astype(int)