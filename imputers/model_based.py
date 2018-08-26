from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np

class ModelBasedImputer(BaseEstimator, TransformerMixin):
    """ Imputes blanks using specified regressor fitted on the present values. """

    def __init__(self, columns, model):
        self.columns = columns
        self.model = model


    def fit(self, X, y, **fit_params):
        self.regressors = {col: clone(self.model) for col in self.columns if X[col].isnull().any()}
        for col, regressor in self.regressors.items():
            y2 = X[col].dropna()
            X2 = X.drop(col, axis=1).loc[y2.index].fillna(0)
            regressor.fit(X2, y2)
        return self


    def transform(self, X):
        for col, regressor in self.regressors.items():
            for i, row in X[X[col].isnull()].iterrows():
                X[col][i] = regressor.predict(np.array([row.drop(col).fillna(0)]))[0]
        return X