from nltk import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from imputers import FillNaTransformer, HotDeckFullImputer, ModelBasedFullImputer
from transformers import BoxCoxTransformer, PolynomialsAdder, CustomBinner, OutliersClipper, FeatureProduct

best = {
#     0
'XGBRegressor': {'params': {'binner2': None,
   'boxcox': BoxCoxTransformer(lambdas_per_column={'age': 2, 'tax': 0, 'lstat': 0}),
   'clipper': OutliersClipper(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'lstat', 'dis']),
   'combinations': None,
   'dropper__drop': [],
   'main_imputer': ModelBasedFullImputer(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
              model=DecisionTreeRegressor( max_depth=None)),
   'poly': PolynomialsAdder(powers_per_column={'crim': [2, 3], 'zn': [2, 3], 'nox': [2, 3], 'indus': [2, 3], 'rm': [2, 3], 'age': [2, 3], 'tax': [2, 3], 'ptratio': [2, 3], 'b': [2, 3], 'lstat': [2, 3], 'dis': [2, 3]}),
   'predictor': XGBRegressor(colsample_bytree=0.6, learning_rate=0.05,
          max_depth=6, n_estimators=500),
   'reduce_dim': None,
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[],
            median=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
            nan_flag=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
            zero=[])},
  'score': 3.7821358047127682,
  'std': 0.4967512627490983},

# 0
'Lasso': {'params': {'binner2': None,
   'boxcox': BoxCoxTransformer(lambdas_per_column={'age': 2, 'tax': 0, 'lstat': 0}),
   'clipper': OutliersClipper(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'lstat', 'dis']),
   'combinations': FeatureProduct(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'lstat', 'dis']),
   'dropper__drop': ['crim_nan','zn_nan','nox_nan','indus_nan','rm_nan','age_nan','tax_nan','ptratio_nan','b_nan','dis_nan'],
   'main_imputer': ModelBasedFullImputer(columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
              model=DecisionTreeRegressor(max_depth=None)),
   'poly': None,
   'predictor': Lasso(alpha=0.01),
   'reduce_dim': None,
   'scaler': RobustScaler(),
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[],
            median=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
            nan_flag=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
            zero=[])},
  'score': 3.993797473454735,
  'std': 0.5808956921355953},

#     0
'LinearRegression': {'params': {'binner2': None,
    'boxcox': BoxCoxTransformer(lambdas_per_column={'age': 2, 'tax': 0, 'lstat': 0}),
    'clipper': OutliersClipper(
        columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b',
                 'lstat', 'dis']),
    'combinations': None,
    'dropper__drop': ['crim_nan','zn_nan','nox_nan','indus_nan','rm_nan','age_nan','tax_nan','ptratio_nan','b_nan','dis_nan'],
    'main_imputer': ModelBasedFullImputer(
        columns=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b','dis'],
        model=DecisionTreeRegressor(max_depth=8)),
    'poly': PolynomialsAdder(
        powers_per_column={'crim': [2, 3], 'zn': [2, 3], 'nox': [2, 3], 'indus': [2, 3],
                           'rm': [2, 3], 'age': [2, 3], 'tax': [2, 3],
                           'ptratio': [2, 3], 'b': [2, 3], 'lstat': [2, 3],
                           'dis': [2, 3]}),
    'predictor': LinearRegression(),
    'reduce_dim': None,
    'scaler': None,
    'simple_imputer': FillNaTransformer(from_dict={}, mean=[],
                                        median=['crim', 'zn', 'nox', 'indus', 'rm','age', 'tax', 'ptratio', 'b', 'dis'],
                                        nan_flag=['crim', 'zn', 'nox', 'indus', 'rm','age', 'tax', 'ptratio', 'b', 'dis'],
                                        zero=[])},
     'score': 4.514645815970899,
     'std': 0.7631593234069367},
}

baseline = {'DecisionTreeRegressor': {'params': {'predictor': DecisionTreeRegressor(criterion='mse', max_depth=4, max_features=None,
              max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_impurity_split=None, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              presort=False, random_state=None, splitter='best'),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[], nan_flag=[],
            zero=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'])},
  'score': 5.5088106991425985,
  'std': 0.293662905734789},
 'KNeighborsRegressor': {'params': {'predictor': KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
             metric_params=None, n_jobs=1, n_neighbors=7, p=2,
             weights='uniform'),
   'scaler': RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
          with_scaling=True),
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
            median=[], nan_flag=[], zero=[])},
  'score': 5.859771905373064,
  'std': 0.90721907618626},
 'LinearRegression': {'params': {'predictor': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
   'scaler': RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
          with_scaling=True),
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[],
            median=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
            nan_flag=[], zero=[])},
  'score': 5.494688479501426,
  'std': 0.5377531716144219}}

advanced_baseline = {
    'XGBRegressor': {'params': {
        'predictor': XGBRegressor(max_depth=4),
        'scaler': None,
        'simple_imputer': FillNaTransformer(from_dict={},
                mean=['crim', 'zn', 'nox', 'indus', 'rm', 'age', 'tax', 'ptratio', 'b', 'dis'],
                median=[], nan_flag=[], zero=[])},
  'score': 4.088217989236429,
  'std': 0.5303490714753816,
},
'XGBRegressor_tuned': {'params': {
    'predictor': XGBRegressor(colsample_bytree=0.6,
                              learning_rate=0.05,
                              max_depth=6,
                              n_estimators=500),
             'scaler': None,
             'simple_imputer': FillNaTransformer(from_dict={},
                                                 mean=['crim', 'zn', 'nox', 'indus', 'rm',
                                                       'age', 'tax', 'ptratio', 'b', 'dis'],
                                                 median=[], nan_flag=[], zero=[])},
  'score': 3.942697483564859,
  'std': 0.6029251513214098}
}
