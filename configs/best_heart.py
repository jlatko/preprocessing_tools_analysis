from nltk import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from imputers import FillNaTransformer, HotDeckFullImputer, ModelBasedFullImputer
from transformers import BoxCoxTransformer, PolynomialsAdder, CustomBinner, OutliersClipper, FeatureProduct

baseline = {'DecisionTreeClassifier': {'params': {'predictor': DecisionTreeClassifier(max_depth=None),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['trestbps', 'chol', 'thalach', 'oldpeak'], median=[],
            nan_flag=[], zero=[])},
  'score': 0.1974390243902439,
  'std': 0.0756691348271984},
 'KNeighborsClassifier': {'params': {'predictor': KNeighborsClassifier(n_neighbors=7),
   'scaler': RobustScaler(),
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[],
            median=['trestbps', 'chol', 'thalach', 'oldpeak'], nan_flag=[],
            zero=[])},
  'score': 0.1924390243902439,
  'std': 0.04087385740197896},
 'LogisticRegression': {'params': {'predictor': LogisticRegression(),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={},
            mean=['trestbps', 'chol', 'thalach', 'oldpeak'], median=[],
            nan_flag=[], zero=[])},
  'score': 0.1825609756097561,
  'std': 0.04141604180434009}}

advanced_baseline = {
    'XGBClassifier': {'params': {'predictor': XGBClassifier(
          colsample_bytree=0.8, learning_rate=0.07,
          max_depth=7, n_estimators=200,),
   'scaler': None,
   'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[], nan_flag=[],
            zero=['trestbps', 'chol', 'thalach', 'oldpeak'])},
  'score': 0.16743902439024388,
  'std': 0.04176646782554455}}

best = {
    'DecisionTreeClassifier': {'params':
        {
            'binner2': CustomBinner(configuration={'chol': {'bins': 3}, 'thalach': {'bins': 3}, 'oldpeak': {'bins': 3}, 'trestbps': {'bins': 3}, 'age': {'bins': 3}, 'slope': {'bins': 3}, 'ca': {'bins': 3}}),
           'boxcox': None,
           'clipper': OutliersClipper(columns=['chol', 'thalach', 'oldpeak', 'trestbps']),
           'combinations': FeatureProduct(columns=['chol', 'thalach', 'oldpeak', 'trestbps']),
           'dropper__drop': ['trestbps_nan', 'chol_nan', 'thalach_nan', 'oldpeak_nan'],
           'main_imputer': HotDeckFullImputer(col_k_pairs=[('trestbps', None), ('chol', None), ('thalach', None), ('oldpeak', None)],
                     default_k=7),
           'poly': None,
           'predictor': DecisionTreeClassifier(max_depth=4),
           'reduce_dim': None,
           'scaler': None,
           'simple_imputer': FillNaTransformer(from_dict={}, mean=[], median=[],
                    nan_flag=['trestbps', 'chol', 'thalach', 'oldpeak'],
                    zero=['trestbps', 'chol', 'thalach', 'oldpeak'])
        },
        'score': 0.14780487804878048,
        'std': 0.03090350740255695},

    'LogisticRegression': {'params':
        {
            'binner2': None,
            'boxcox': BoxCoxTransformer(lambdas_per_column={'chol': 0, 'thalach': 2, 'trestbps': 0}),
            'clipper': OutliersClipper(columns=['chol', 'thalach', 'oldpeak', 'trestbps']),
            'combinations': None,
            'dropper__drop': [],
            'main_imputer': ModelBasedFullImputer(columns=['trestbps', 'chol', 'thalach', 'oldpeak'],
                      model=LinearRegression()),
            'poly': PolynomialsAdder(powers_per_column={'chol': [2], 'thalach': [2], 'oldpeak': [2], 'trestbps': [2]}),
            'predictor': LogisticRegression(),
            'reduce_dim': PCA(n_components=10),
            'scaler': None,
            'simple_imputer': FillNaTransformer(from_dict={},
                    mean=['trestbps', 'chol', 'thalach', 'oldpeak'], median=[],
                    nan_flag=['trestbps', 'chol', 'thalach', 'oldpeak'], zero=[])},
        'score': 0.14280487804878048,
        'std': 0.03915450868377355},
    'XGBClassifier': {'params':
        {
            'binner2': CustomBinner(configuration={'chol': {'bins': 3}, 'thalach': {'bins': 3}, 'oldpeak': {'bins': 3}, 'trestbps': {'bins': 3}, 'age': {'bins': 3}, 'slope': {'bins': 3}, 'ca': {'bins': 3}},
                  drop=False, nan=False),
           'boxcox': BoxCoxTransformer(lambdas_per_column={'chol': 0, 'thalach': 2, 'trestbps': 0}),
           'clipper': OutliersClipper(columns=['chol', 'thalach', 'oldpeak', 'trestbps']),
           'combinations': None,
           'dropper__drop': ['trestbps_nan', 'chol_nan', 'thalach_nan', 'oldpeak_nan'],
           'main_imputer': HotDeckFullImputer(col_k_pairs=[('trestbps', None), ('chol', None), ('thalach', None), ('oldpeak', None)],
                     default_k=7),
           'poly': None,
           'predictor': XGBClassifier(colsample_bytree=0.8, learning_rate=0.07,
                  max_depth=7, n_estimators=200,),
           'reduce_dim': SelectFromModel(estimator=LogisticRegression(C=0.999, penalty='l1')),
           'scaler': None,
           'simple_imputer': FillNaTransformer(from_dict={},
                    mean=['trestbps', 'chol', 'thalach', 'oldpeak'], median=[],
                    nan_flag=['trestbps', 'chol', 'thalach', 'oldpeak'], zero=[])},
  'score': 0.15243902439024387,
  'std': 0.04655758333858798}
}