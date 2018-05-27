from sklearn.base import BaseEstimator, TransformerMixin

class FixSkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.select_dtypes(exclude = ["object"]).columns
        skewness = X[self.columns].apply(lambda x: skew(x))
        skewness = skewness[abs(skewness)>0.5]
        self.skew_features = skewness.index
        return self

    def transform(self, X):
        X[self.skew_features] = np.log1p(X[self.skew_features])
        return X