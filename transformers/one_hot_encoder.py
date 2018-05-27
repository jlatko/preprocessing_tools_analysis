from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    # numerical - numerical columns to be treated as categorical
    # columns - columns to use (if None then all categorical variables are included)
    def __init__(self, columns=None, numerical=[]):
        self.numerical = numerical
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        # if none specified – get all non numerical columns
        if self.columns == None:
            self.columns = X.select_dtypes(include = ["object"]).columns.tolist()
        self.columns += self.numerical
        # get all possible column values to filter not seen values
        self.allowed_columns = [ "{}_{}".format(column, val) for column in self.columns for val in X[column].unique() ]
        return self

    def transform(self, X, y=None, **fit_params):
        # cast numerical columns to strings
        for col in X[self.columns].select_dtypes(exclude = ["object"]).columns:
            X[col] = X[col].astype('str')
        one_hots = pd.get_dummies(X[self.columns], prefix=self.columns)
        missing_cols = set(self.allowed_columns) - set(one_hots.columns)
        for c in missing_cols:
            one_hots[c] = 0
        return pd.concat([X.drop(self.columns, axis=1), one_hots.filter(self.allowed_columns)], axis=1)