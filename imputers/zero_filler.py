from sklearn.base import BaseEstimator, TransformerMixin

class ZeroFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X.fillna(0)