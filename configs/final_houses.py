from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from diagnostics.evaluation import rmse
from imputers import FillNaTransformer, HotDeckFullImputer, ModelBasedFullImputer, ZeroFiller
from tools.datasets import get_houses
from transformers import BoxCoxTransformer, PolynomialsAdder, CustomBinner, OutliersClipper, FeatureProduct, \
    CustomBinaryBinner, FeatureDropper, CustomOneHotEncoder


def get_test_config_houses():
    data, labels, continuous, discrete, dummy, categorical, target, missing = get_houses(test=False)
    test_data, test_labels = get_houses(test=True)[0:2]
    train = data.drop(target, axis=1)
    test = test_data.drop(target, axis=1)
    scorer = rmse
    binner = CustomBinaryBinner({ col: {'values': [train[col].max()]} for col in continuous + discrete })

    one_hot = CustomOneHotEncoder(columns=categorical)
    model = Pipeline([
        ('onehot', one_hot),
        ('clipper', None),
        ('binner', None),
        ('binner2', None),
        ('simple_imputer', None),
        ('zero_filler', ZeroFiller()),
        ('main_imputer', None),
        ('dropper', FeatureDropper(drop=[])),
        ('poly', None),
        ('combinations', None),
        ('boxcox', None),
        ('scaler', None),
        ('reduce_dim', None),
        ('predictor', None)
    ])
    params = {
        'LinearRegression_best': {
                'params': {'binner2': binner,
               'boxcox': BoxCoxTransformer(lambdas_per_column={'BsmtFinSF1': 0, 'LowQualFinSF': 0, 'GrLivArea': 0, 'WoodDeckSF': 0}),
               'clipper': None,
               'combinations': FeatureProduct(columns=['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']),
               'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
               'main_imputer': ModelBasedFullImputer(columns=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'],
                          model=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=7, normalize=False)),
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
          'std': 1575.0296698711768},
     #    same as baseline
     'XGBRegressor_best': {'params': {'binner2': None,
       'boxcox': None,
       'clipper': None,
       'combinations': None,
       'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
       'main_imputer': None,
       'poly': None,
       'predictor': XGBRegressor(
              colsample_bytree=1, learning_rate=0.07,
              max_depth=3, n_estimators=1000, n_jobs=7),
       'reduce_dim': None,
       'scaler': None,
       'simple_imputer': FillNaTransformer(from_dict={},
                mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
                nan_flag=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], zero=[])},
      'score': 25011.258367120317,
      'std': 1390.9293424644447},
    'DecisionTreeRegressor_base': {'params': {'predictor': DecisionTreeRegressor(),
       'scaler': None,
       'simple_imputer': FillNaTransformer(from_dict={},
                mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
                nan_flag=[], zero=[])},
      'score': 41942.13985251157,
      'std': 4160.532009551353},
     'KNeighborsRegressor_base': {'params': {'predictor': KNeighborsRegressor(n_neighbors=7, n_jobs=7),
       'scaler': None,
       'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[], nan_flag=[],
                zero=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])},
      'score': 49074.08623384354,
      'std': 4863.286944918721},
     'LinearRegression_base': {'params': {'predictor': LinearRegression(n_jobs=7),
       'scaler': None,
       'simple_imputer': FillNaTransformer(from_dict={},
                mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
                nan_flag=[], zero=[])},
      'score': 58196.855144405956,
      'std': 28243.597268210826},

        'XGBRegressor_base': {'params': {
        'predictor': XGBRegressor(max_depth=3, n_jobs=7),
       'scaler': None,
       'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[], nan_flag=[],
                zero=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])},
      'score': 26368.673265668913,
      'std': 1310.375215389942},
    'XGBRegressor_tuned_base': {'params': {'predictor': XGBRegressor(
              colsample_bytree=1,learning_rate=0.07,
              max_depth=3, n_estimators=1000, n_jobs=7),
       'scaler': None,
       'simple_imputer': FillNaTransformer(from_dict={},
                mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
                nan_flag=[], zero=[])},
      'score': 24894.281304255797,
      'std': 1177.011047285853},

    'DecisionTreeRegressor_best': {'params': {'binner': None,
    'binner2': binner,
    'clipper': OutliersClipper(columns=['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']),
    'combinations': FeatureProduct(columns=['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']),
    'dropper__drop': ['LotFrontage_nan', 'MasVnrArea_nan', 'GarageYrBlt_nan'],
    'main_imputer': HotDeckFullImputer(col_k_pairs=[('LotFrontage', None), ('MasVnrArea', None), ('GarageYrBlt', None)], default_k=5),
    'poly': PolynomialsAdder(powers_per_column={'LotFrontage': [2], 'LotArea': [2], 'MasVnrArea': [2], 'BsmtFinSF1': [2], 'BsmtFinSF2': [2], 'BsmtUnfSF': [2], 'TotalBsmtSF': [2], '1stFlrSF': [2], '2ndFlrSF': [2], 'LowQualFinSF': [2], 'GrLivArea': [2], 'GarageArea': [2], 'WoodDeckSF': [2], 'OpenPorchSF': [2], 'EnclosedPorch': [2], '3SsnPorch': [2], 'ScreenPorch': [2], 'PoolArea': [2], 'MiscVal': [2]}),
    'predictor': DecisionTreeRegressor(),
    'reduce_dim': SelectFromModel(estimator=RandomForestRegressor(max_depth=8)),
    'simple_imputer': FillNaTransformer(from_dict={},
        mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], median=[],
        nan_flag=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], zero=[])},
    'score': 37866.96889728187,
    'std': 5359.25597193946},

    'Lasso_best': {'params': {
        'binner': None,
        'binner2': binner,
        'clipper': None,
        'combinations': FeatureProduct(
            columns=['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']),
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
        'std': 639.0998169991468} ,

        'Lasso_base': {'params': {
            'predictor': Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
                               normalize=False, positive=False, precompute=False, random_state=None,
                               selection='cyclic', tol=0.0001, warm_start=False),
            'scaler': RobustScaler(),
            'simple_imputer': FillNaTransformer(from_dict={},
                                                mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])},
            'score': 0,
            'std': 0}


    }

    return data, test, test_labels, scorer, model, params, target
