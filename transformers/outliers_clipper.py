from sklearn.base import BaseEstimator, TransformerMixin


class OutliersClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        X_mean = X[self.columns].mean()
        X_std = X[self.columns].std()
        self.min_per_col = X_mean - 3 * X_std
        self.max_per_col = X_mean + 3 * X_std
        return self

    def transform(self, X):
        X[self.columns] = X[self.columns].clip(self.min_per_col, self.max_per_col, axis=1)
        return X
