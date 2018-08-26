from nltk import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from imputers import FillNaTransformer, HotDeckFullImputer, ModelBasedFullImputer
from transformers import BoxCoxTransformer, PolynomialsAdder, CustomBinner, OutliersClipper, FeatureProduct, \
    CustomBinaryBinner

best = {
    'LinearRegression': {
            'params': {'binner2': CustomBinaryBinner(),
           'boxcox': BoxCoxTransformer(lambdas_per_column={'BsmtFinSF1': 0, 'LowQualFinSF': 0, 'GrLivArea': 0, 'WoodDeckSF': 0}),
           'clipper': None,
           'combinations': FeatureProduct(columns=['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']),
           'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
           'main_imputer': ModelBasedFullImputer(columns=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'],
                      model=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)),
           'poly': PolynomialsAdder(powers_per_column={'LotFrontage': [3], 'BsmtFinSF1': [3], 'MasVnrArea': [3], '1stFlrSF': [3], 'GarageArea': [3], 'TotalBsmtSF': [3], 'GrLivArea': [3]}),
           'predictor': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
           'reduce_dim': PCA(copy=True, iterated_power='auto', n_components=80, random_state=None,
             svd_solver='auto', tol=0.0, whiten=False),
           'scaler': RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
                  with_scaling=True),
           'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[],
                    nan_flag=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'],
                    zero=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])},
      'score': 23615.61547841841,
      'std': 1575.0296698711768
},
 #    same as baseline
 'XGBRegressor': {'params': {'binner2': None,
   'boxcox': None,
   'clipper': None,
   'combinations': None,
   'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
   'main_imputer': None,
   'poly': None,
   'predictor': XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
          colsample_bytree=1, gamma=0, learning_rate=0.07, max_delta_step=0,
          max_depth=3, min_child_weight=1, n_estimators=1000,
          n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
          reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
          silent=True, subsample=1),
   'reduce_dim': None,
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
            nan_flag=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], zero=[])},
  'score': 25011.258367120317,
  'std': 1390.9293424644447}
}
baseline = {'DecisionTreeRegressor': {'params': {'predictor': DecisionTreeRegressor(),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
            nan_flag=[], zero=[])},
  'score': 41942.13985251157,
  'std': 4160.532009551353},
 'KNeighborsRegressor': {'params': {'predictor': KNeighborsRegressor(n_neighbors=7),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[], nan_flag=[],
            zero=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])},
  'score': 49074.08623384354,
  'std': 4863.286944918721},
 'LinearRegression': {'params': {'predictor': LinearRegression(),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
            nan_flag=[], zero=[])},
  'score': 58196.855144405956,
  'std': 28243.597268210826}}

advanced_baseline =  {
    'XGBRegressor': {'params': {
    'predictor': XGBRegressor(max_depth=3),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[], nan_flag=[],
            zero=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])},
  'score': 26368.673265668913,
  'std': 1310.375215389942},
'XGBRegressor_tuned': {'params': {'predictor': XGBRegressor(
          colsample_bytree=1,learning_rate=0.07,
          max_depth=3, n_estimators=1000),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
            nan_flag=[], zero=[])},
  'score': 24894.281304255797,
  'std': 1177.011047285853}
}

best2 = {'DecisionTreeRegressor': {'params': {'binner': None,
   'binner2': CustomBinaryBinner(configuration={'LotFrontage': {'values': [182.0]}, 'LotArea': {'values': [215245]}, 'MasVnrArea': {'values': [1378.0]}, 'BsmtFinSF1': {'values': [2188]}, 'BsmtFinSF2': {'values': [1120]}, 'BsmtUnfSF': {'values': [2336]}, 'TotalBsmtSF': {'values': [3206]}, '1stFlrSF': {'values': [3228]}, '2ndFlrSF': ... [2010.0]}, 'GarageCars': {'values': [4]}, 'MoSold': {'values': [12]}, 'YrSold': {'values': [2010]}},
             drop=False, nan=False),
   'clipper': OutliersClipper(columns=['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']),
   'combinations': FeatureProduct(columns=['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']),
   'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
   'main_imputer': HotDeckFullImputer(col_k_pairs=[('LotFrontage', None), ('MasVnrArea', None), ('GarageYrBlt', None)],
             default_k=5),
   'poly': PolynomialsAdder(powers_per_column={'LotFrontage': [2], 'LotArea': [2], 'MasVnrArea': [2], 'BsmtFinSF1': [2], 'BsmtFinSF2': [2], 'BsmtUnfSF': [2], 'TotalBsmtSF': [2], '1stFlrSF': [2], '2ndFlrSF': [2], 'LowQualFinSF': [2], 'GrLivArea': [2], 'GarageArea': [2], 'WoodDeckSF': [2], 'OpenPorchSF': [2], 'EnclosedPorch': [2], '3SsnPorch': [2], 'ScreenPorch': [2], 'PoolArea': [2], 'MiscVal': [2]}),
   'predictor': DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
              max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_impurity_split=None, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              presort=False, random_state=None, splitter='best'),
   'reduce_dim': SelectFromModel(estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=8,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
              oob_score=False, random_state=None, verbose=0, warm_start=False),
           norm_order=1, prefit=False, threshold=None),
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
            nan_flag=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], zero=[])},
  'score': 37866.96889728187,
  'std': 5359.25597193946},
 'Lasso': {'params': {'binner': None,
   'binner2': CustomBinaryBinner(configuration={'LotFrontage': {'values': [182.0]}, 'LotArea': {'values': [215245]}, 'MasVnrArea': {'values': [1378.0]}, 'BsmtFinSF1': {'values': [2188]}, 'BsmtFinSF2': {'values': [1120]}, 'BsmtUnfSF': {'values': [2336]}, 'TotalBsmtSF': {'values': [3206]}, '1stFlrSF': {'values': [3228]}, '2ndFlrSF': ... [2010.0]}, 'GarageCars': {'values': [4]}, 'MoSold': {'values': [12]}, 'YrSold': {'values': [2010]}},
             drop=False, nan=False),
   'clipper': None,
   'combinations': FeatureProduct(columns=['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']),
   'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
   'main_imputer': HotDeckFullImputer(col_k_pairs=[('LotFrontage', None), ('MasVnrArea', None), ('GarageYrBlt', None)],
             default_k=7),
   'poly': None,
   'predictor': Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, positive=False, precompute=False, random_state=None,
      selection='cyclic', tol=0.0001, warm_start=False),
   'reduce_dim': PCA(copy=True, iterated_power='auto', n_components=80, random_state=None,
     svd_solver='auto', tol=0.0, whiten=False),
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
            nan_flag=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], zero=[])},
  'score': 24138.931080813963,
  'std': 639.0998169991468}}