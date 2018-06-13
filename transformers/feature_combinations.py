from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProduct(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        for i in range(len(self.columns)):
            for j in range(i+1, len(self.columns)):
                col_i = self.columns[i]
                col_j = self.columns[j]
                X[col_i + '*' + col_j] = X[col_i] * X[col_j]
        return X