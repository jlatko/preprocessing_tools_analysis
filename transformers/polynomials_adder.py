from sklearn.base import BaseEstimator, TransformerMixin


class PolynomialsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, powers_per_column):
        self.powers_per_column = powers_per_column

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        for col, powers in self.powers_per_column.items():
            for power in powers:
                X[col+"^"+str(power)] = X[col]**power
        return X