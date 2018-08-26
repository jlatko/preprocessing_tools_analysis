from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor

from diagnostics.evaluation import rev_weighted_quad_kappa
from imputers import FillNaTransformer, HotDeckFullImputer, ModelBasedFullImputer, Pipeline
from tools.datasets import get_prudential
from transformers import BoxCoxTransformer, PolynomialsAdder, CustomBinner, OutliersClipper, FeatureProduct, \
    LabelsClipper, CustomBinaryBinner, FeatureDropper, CustomOneHotEncoder


def get_test_config_prudential():
    data, labels, continuous, discrete, dummy, categorical, target, missing = get_prudential(test=False)
    test_data, test_labels = get_prudential(test=True)[0:2]
    test = test_data.drop(target, axis=1)
    scorer = rev_weighted_quad_kappa

    one_hot = CustomOneHotEncoder(columns=categorical)
    model = Pipeline([
        ('onehot', one_hot),
        ('clipper', None),
        ('binner', None),
        ('binner2', None),
        ('simple_imputer', None),
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
     #    'DecisionTreeClassifier_base': {'params': {'predictor': LabelsClipper(regressor=DecisionTreeClassifier()),
     #   'scaler': RobustScaler(),
     #   'simple_imputer': FillNaTransformer(from_dict={}, mean=[],
     #            median=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
     #            nan_flag=[], zero=[])},
     #  'score': 0.5894217150410359,
     #  'std': 0.007355958627832262},
     # 'DecisionTreeRegressor': {'params': {'predictor': LabelsClipper(regressor=DecisionTreeRegressor()),
     #   'scaler': None,
     #   'simple_imputer': FillNaTransformer(from_dict={},
     #            mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
     #            median=[], nan_flag=[], zero=[])},
     #  'score': 0.5686751743396636,
     # #  'std': 0.0060008388054878},
     # 'LinearRegression': {'params': {'predictor': LabelsClipper(regressor=LinearRegression()),
     #   'scaler': RobustScaler(),
     #   'simple_imputer': FillNaTransformer(from_dict={},
     #            mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
     #            median=[], nan_flag=[], zero=[])},
     #  'score': 0.45392704971818426,
     #  'std': 0.003043510387390882},

    # 'XGBRegressor_base': {'params': {
    #     'predictor': LabelsClipper(regressor=XGBRegressor(max_depth=8)),
    #     'scaler': RobustScaler(),
    #     'simple_imputer': FillNaTransformer(from_dict={},
    #             mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
    #             median=[], nan_flag=[], zero=[])},
    #     'score': 0.40115420082450576,
    #     'std': 0.00512923799723226},

    'XGBRegressor_tuned_base': {'params':
        {'predictor': LabelsClipper(regressor=XGBRegressor(n_jobs=7,
              colsample_bytree=1,learning_rate=0.07,
              max_depth=16, n_estimators=1000)),
       'scaler': RobustScaler(),
       'simple_imputer': FillNaTransformer(from_dict={},
                mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
                median=[], nan_flag=[], zero=[])},
      'score': 0.397178540923399,
      'std': 0.0052540614137706766},

    # 'LinearRegression_best': {'params': {'binner': CustomBinner(configuration={'Medical_History_1': {'values': [0, 240], 'bins': [0, 1, 10, 30, 100, 200, 240]}, 'Medical_History_10': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_15': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_24': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_32': {'values': [0, 240], 'bins': [0, 1, 2, 240]}},
    #           drop=False, nan=False),
    #    'binner2': CustomBinaryBinner(configuration={'Product_Info_4': {'bins': 7}, 'Ins_Age': {'bins': 7}, 'Ht': {'bins': 7}, 'Wt': {'bins': 7}, 'BMI': {'bins': 7}, 'Employment_Info_1': {'bins': 7}, 'Employment_Info_4': {'bins': 7}, 'Employment_Info_6': {'bins': 7}, 'Insurance_History_5': {'bins': 7}, 'Family_Hist_2': {'bins': 7}, 'Family_Hist_3': {'bins': 7}, 'Family_Hist_4': {'bins': 7}, 'Family_Hist_5': {'bins': 7}},
    #              drop=False, nan=False),
    #    'boxcox': BoxCoxTransformer(lambdas_per_column={'Wt': 0.5}),
    #    'combinations': FeatureProduct(columns=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']),
    #    'dropper__drop': [],
    #    'main_imputer': ModelBasedFullImputer(columns=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
    #               model=XGBRegressor()),
    #    'poly': PolynomialsAdder(powers_per_column={'Product_Info_4': [2, 3], 'Ins_Age': [2, 3], 'Ht': [2, 3], 'Wt': [2, 3], 'BMI': [2, 3], 'Employment_Info_1': [2, 3], 'Employment_Info_4': [2, 3], 'Employment_Info_6': [2, 3], 'Insurance_History_5': [2, 3], 'Family_Hist_2': [2, 3], 'Family_Hist_3': [2, 3], 'Family_Hist_4': [2, 3], 'Family_Hist_5': [2, 3]}),
    #    'predictor': LabelsClipper(regressor=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=False)),
    #    'reduce_dim': SelectFromModel(estimator=Lasso(alpha=0.0001)),
    #    'scaler': RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
    #           with_scaling=True),
    #    'simple_imputer': FillNaTransformer(from_dict={},
    #             mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
    #             median=[],
    #             nan_flag=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
    #             zero=[])},
    #   'score': 0.4226586420535899,
    #   'std': 0.005930880816785933},
     'XGBRegressor_best': {'params': {'binner': CustomBinner(configuration={'Medical_History_1': {'values': [0, 240], 'bins': [0, 1, 10, 30, 100, 200, 240]}, 'Medical_History_10': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_15': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_24': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_32': {'values': [0, 240], 'bins': [0, 1, 2, 240]}},
              drop=False, nan=False),
       'binner2': None,
       'boxcox': BoxCoxTransformer(lambdas_per_column={'Wt': 0.5}),
       'combinations': FeatureProduct(columns=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']),
       'dropper__drop': ['Product_Info_4_nan','Ins_Age_nan','Ht_nan','Wt_nan','BMI_nan','Employment_Info_1_nan','Employment_Info_4_nan','Employment_Info_6_nan','Insurance_History_5_nan','Family_Hist_2_nan','Family_Hist_3_nan','Family_Hist_4_nan','Family_Hist_5_nan','Medical_History_1_nan','Medical_History_10_nan','Medical_History_15_nan','Medical_History_24_nan','Medical_History_32_nan'],
       'main_imputer': HotDeckFullImputer(col_k_pairs=[('Product_Info_4', None), ('Ins_Age', None), ('Ht', None), ('Wt', None), ('BMI', None), ('Employment_Info_1', None), ('Employment_Info_4', None), ('Employment_Info_6', None), ('Insurance_History_5', None), ('Family_Hist_2', None), ('Family_Hist_3', None), ('Family_Hist_4', None), ('Family_Hist_5', None), ('Medical_History_1', None), ('Medical_History_10', None), ('Medical_History_15', None), ('Medical_History_24', None), ('Medical_History_32', None)],
                 default_k=5),
       'poly': None,
       'predictor': LabelsClipper(regressor=XGBRegressor(n_jobs=7,
              max_depth=8)),
       'reduce_dim': SelectFromModel(estimator=Lasso(alpha=0.0001)),
       'scaler': RobustScaler(),
       'simple_imputer': FillNaTransformer(from_dict={},
                mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
                median=[],
                nan_flag=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
                zero=[])},
      'score': 0.4013165044558284,
      'std': 0.005226847144375746},
      # 'XGBRegressor_tuned_best': {'params': {'binner': CustomBinner(configuration={'Medical_History_1': {'values': [0, 240], 'bins': [0, 1, 10, 30, 100, 200, 240]}, 'Medical_History_10': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_15': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_24': {'values': [0, 240], 'bins': [0, 1, 2, 240]}, 'Medical_History_32': {'values': [0, 240], 'bins': [0, 1, 2, 240]}},
      #         drop=False, nan=False),
      #  'binner2': None,
      #  'boxcox': BoxCoxTransformer(lambdas_per_column={'Wt': 0.5}),
      #  'combinations': FeatureProduct(columns=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']),
      #  'dropper__drop': ['Product_Info_4_nan','Ins_Age_nan','Ht_nan','Wt_nan','BMI_nan','Employment_Info_1_nan','Employment_Info_4_nan','Employment_Info_6_nan','Insurance_History_5_nan','Family_Hist_2_nan','Family_Hist_3_nan','Family_Hist_4_nan','Family_Hist_5_nan','Medical_History_1_nan','Medical_History_10_nan','Medical_History_15_nan','Medical_History_24_nan','Medical_History_32_nan'],
      #  'main_imputer': HotDeckFullImputer(col_k_pairs=[('Product_Info_4', None), ('Ins_Age', None), ('Ht', None), ('Wt', None), ('BMI', None), ('Employment_Info_1', None), ('Employment_Info_4', None), ('Employment_Info_6', None), ('Insurance_History_5', None), ('Family_Hist_2', None), ('Family_Hist_3', None), ('Family_Hist_4', None), ('Family_Hist_5', None), ('Medical_History_1', None), ('Medical_History_10', None), ('Medical_History_15', None), ('Medical_History_24', None), ('Medical_History_32', None)],
      #            default_k=5),
      #  'poly': None,
      #  'predictor': LabelsClipper(regressor=XGBRegressor(
      #         colsample_bytree=1,learning_rate=0.07,
      #         max_depth=16, n_estimators=1000)),
      #  'reduce_dim': SelectFromModel(estimator=Lasso(alpha=0.0001)),
      #  'scaler': RobustScaler(),
      #  'simple_imputer': FillNaTransformer(from_dict={},
      #           mean=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
      #           median=[],
      #           nan_flag=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'],
      #           zero=[])},
      # 'score': 0.4013165044558284,
      # 'std': 0.005226847144375746}
              }

    return data, test, test_labels, scorer, model, params, target