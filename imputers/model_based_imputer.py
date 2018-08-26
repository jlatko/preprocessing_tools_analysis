from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline


class ModelBasedColImputer(BaseEstimator, TransformerMixin):
    """
    Imputes blanks in one column using specified regressor fitted on the present values.
    Uses colname_nan to indicate which rows were nan
    """

    def __init__(self, column, model):
        self.column = column
        self.model = clone(model)


    def fit(self, X, y=None, **fit_params):
        without_na = X[~X[self.column + '_nan'].astype('bool')]
        without_target_col = without_na.drop([self.column, self.column + '_nan'], axis=1)
        self.model.fit(without_target_col, without_na[self.column])
        return self


    def transform(self, X):
        with_na = X[X[self.column + '_nan'].astype('bool')]
        without_target_col = with_na.drop([self.column, self.column + '_nan'], axis=1)
        if without_target_col.shape[0]:
            X.loc[X[self.column + '_nan'].astype('bool'), self.column] = self.model.predict(without_target_col)
        return X


class ModelBasedFullImputer(BaseEstimator, TransformerMixin):
    """ Imputes blanks in multiple columns by combining several ModelBasedColImputers. """

    def __init__(self, columns, model):
        self.columns = columns
        self.model = model


    def fit(self, X, y=None, **fit_params):
        imputers = [
            (col + '_imputer', ModelBasedColImputer(column=col, model=self.model))
            for col in self.columns
            if col + '_nan' in X.columns and ( 0 < X[col + '_nan'].sum() < X.shape[0]) ]
        self.pipe = Pipeline(imputers)
        return self.pipe.fit(X, y)


    def transform(self, X):
        return self.pipe.transform(X)
