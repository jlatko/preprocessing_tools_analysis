from sklearn.base import BaseEstimator, TransformerMixin

class FillNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mean=[], median=[], zero=[], nan_flag=[], from_dict={}):
        self.median = median
        self.mean = mean
        self.zero = zero
        self.nan_flag = nan_flag
        self.from_dict = from_dict

    def fit(self, X, y=None, **fit_params):
        self.median = [col for col, missing in X[self.median].isnull().sum().items() if missing]
        self.mean = [col for col, missing in X[self.mean].isnull().sum().items() if missing]
        self.zero = [col for col, missing in X[self.zero].isnull().sum().items() if missing]
        self.nan_flag = [col for col, missing in X[self.nan_flag].isnull().sum().items() if missing]
        self.from_dict_filtered = [col for col, missing in X[list(self.from_dict.keys())].isnull().sum().items() if missing]
        self.from_dict = {key: val for key, val in self.from_dict.items() if key in self.from_dict_filtered}
        self.train_median = X[self.median].median()
        self.train_mean = X[self.mean].mean()
        return self

    def transform(self, X):
        for col in self.nan_flag:
            X.loc[:, col+'_nan'] = X[col].isnull() * 1
        X.loc[:, self.median] = X[self.median].fillna(self.train_median)
        X.loc[:, self.mean] = X[self.mean].fillna(self.train_mean)
        X.loc[:, self.zero] = X[self.zero].fillna(0)
        for key, val in self.from_dict.items():
            X.loc[:, key] = X[key].fillna(val)
        return X