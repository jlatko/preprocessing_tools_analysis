import json
import pickle
import traceback
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from diagnostics.evaluation import rev_weighted_quad_kappa
from diagnostics.evaluation import fit_and_eval, rev_weighted_quad_kappa, rmse, error_rate
from imputers import *
from tools.datasets import get_boston, get_heart, get_houses, get_prudential
from diagnostics.grid_search import custom_grid_search
from config import *
from transformers import *
import warnings
from visualization.plot_results import plot_and_save_results

""" 
Main script used for performing grid searches and comparing preprocessing methods.
This file was modified numerous times to test different models. 
"""

warnings.filterwarnings('ignore')

# get labels and ticks for plotting and report
def get_class(params):
    model_name =  'baseline'

    if params['clipper']:
        label = 'clipped'
    else:
        label = 'not clipped'
    return model_name, label

def get_class_from_list(setting_list):
    def get_c(params):
        model_name, label = setting_list[get_c.i]
        get_c.i += 1
        print(params)
        print(model_name, label)
        return model_name, label
    get_c.i = 0
    return get_c


TEST_NAME = 'binning'
sets = [
    # 'prudential',
    # 'boston',
    'houses',
    # 'heart'
]
MISSING = True
outliers = [
    0,
    # 'clip',
    # 'remove'
]



model_results = {}
for dataset in sets:
    print("Performing tests for ", dataset)

    #  ---  choose dataset  ---
    if dataset == 'prudential':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_prudential()
        train = data.drop(target, axis=1)
        cv = StratifiedKFold(5)
        scorer = rev_weighted_quad_kappa
        predictors = [
            LabelsClipper(regressor=XGBRegressor(
              colsample_bytree=1,learning_rate=0.1,
              max_depth=4, n_estimators=100, n_jobs=4)),
            LabelsClipper(regressor=LinearRegression(n_jobs=4)),
        ]
        predictor = LabelsClipper(regressor=LinearRegression())
        binner = CustomBinaryBinner({ col: {'bins': 7} for col in continuous })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous },
        #     { col: {'bins': 5} for col in continuous },
        #     { col: {'bins': 7} for col in continuous },
        #     { col: {'values': [train[col].max()]} for col in continuous }]
        BOX_COX = BOX_COX_P
        precision = 4

    elif dataset == 'boston':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_boston(missing=MISSING)
        train = data.drop(target, axis=1)
        cv = KFold(5, shuffle=True, random_state=0)
        scorer = rmse
        predictors = [
            # XGBRegressor(colsample_bytree=0.6,
            #                   learning_rate=0.05,
            #                   max_depth=6,
            #                   n_estimators=500),
            LinearRegression(),
            Lasso(alpha=0.01),
        ]

        predictor = LinearRegression()
        binner = CustomBinner({ col: {'bins': 7} for col in continuous + discrete })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous + discrete },
        #     { col: {'bins': 5} for col in continuous + discrete },
        #     { col: {'bins': 7} for col in continuous + discrete },
        #     { col: {'values': [train[col].max()]} for col in continuous + discrete }]
        BOX_COX = BOX_COX_B
        reducers = [
                None,
                PCA(10),
                SelectFromModel(Lasso(alpha=0.001)),
                # SelectFromModel(Lasso(alpha=0.01)),
                # SelectFromModel(XGBRegressor(max_depth=4)),
            ]
        precision = 3

    elif dataset == 'houses':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_houses()
        cv = KFold(5, shuffle=True, random_state=0)
        scorer = rmse
        precision = 0
        predictors = [
            # XGBRegressor(
            #   colsample_bytree=1,learning_rate=0.07,
            #   max_depth=3, n_estimators=1000),
            # LinearRegression(),
            DecisionTreeRegressor(),
            Lasso()
        ]

        predictor = LinearRegression()
        train = data.drop(target, axis=1)
        binner = CustomBinaryBinner({ col: {'values': [train[col].max()]} for col in continuous + discrete })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous + discrete },
        #     # { col: {'bins': 5} for col in continuous + discrete },
        #     { col: {'bins': 7} for col in continuous + discrete },
        #     { col: {'values': [train[col].max()]} for col in continuous + discrete }]
        top_cont = ['LotFrontage', 'BsmtFinSF1', 'MasVnrArea', '1stFlrSF', 'GarageArea',
            'TotalBsmtSF', 'GrLivArea']
        reducers = [
                None,
                PCA(70),
                PCA(80),
                SelectFromModel(RandomForestRegressor(max_depth=8)),
            ]
        BOX_COX = BOX_COX_HO

    elif dataset == 'heart':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_heart(missing=MISSING)
        train = data.drop(target, axis=1)
        cv = KFold(5, shuffle=True, random_state=0)
        scorer = error_rate
        predictors = [
            XGBClassifier(
              colsample_bytree=0.8, learning_rate=0.07,
              max_depth=7, n_estimators=200),
            LogisticRegression(),
            DecisionTreeClassifier(max_depth=4)
        ]

        predictor = LogisticRegression()
        reducers = [
                None,
                PCA(10),
                SelectFromModel(LogisticRegression(penalty='l1', C=0.999))
            ]
        # BINNER_CONFIG = { col: {'bins': 3} for col in continuous + discrete }
        binner = CustomBinner({ col: {'bins': 3} for col in continuous + discrete })
        BOX_COX = BOX_COX_HE
        precision = 4

    else:
        continue


    #  ---  initialize pipeline  ---
    one_hot = CustomOneHotEncoder(columns=categorical) if dataset != 'boston' else None
    model = Pipeline([
        ('onehot', one_hot),
        ('clipper', OutliersClipper(continuous)),
        ('binner', CustomBinner(BINNER_CONFIG_PRUD)),
        ('binner2', binner),
        ('simple_imputer', FillNaTransformer()),
        # ('zero_filler', ZeroFiller()),  # just in case there are any left
        ('main_imputer', HotDeckFullImputer(col_k_pairs={})),
        ('dropper', FeatureDropper(drop=[])),
        ('poly', PolynomialsAdder(powers_per_column={col: [2] for col in continuous})),
        ('combinations', FeatureProduct(columns=continuous)),
        ('boxcox', BoxCoxTransformer(BOX_COX)),
        ('scaler', RobustScaler()),
        ('reduce_dim', None),
        ('predictor', None)
    ])

    params =[ {
            'clipper': [
                OutliersClipper(continuous),
                None
            ],
            'binner': [
                None,
                # CustomBinner(BINNER_CONFIG_PRUD)
            ],
            'binner2': [
                None,
                binner
            ],
            'simple_imputer': [
                FillNaTransformer(mean=missing, nan_flag=missing),
            ],
            'main_imputer': [
                None,
                HotDeckFullImputer(col_k_pairs=[(col, None) for col in missing], default_k=3),
                HotDeckFullImputer(col_k_pairs=[(col, None) for col in missing], default_k=5),
                HotDeckFullImputer(col_k_pairs=[(col, None) for col in missing], default_k=7),
                ModelBasedFullImputer(columns=missing, model=DecisionTreeRegressor()),
                ModelBasedFullImputer(columns=missing, model=RandomForestRegressor()),
                ModelBasedFullImputer(columns=missing, model=LinearRegression()),
                ModelBasedFullImputer(columns=missing, model=KNeighborsRegressor(n_neighbors=5)),
            ],
            'poly': [
                None,
                PolynomialsAdder(powers_per_column={col: [2] for col in continuous})
            ],
            'combinations': [
                None,
                FeatureProduct(columns=top_cont),
            ],
            'dropper__drop': [
                [],
                [col + '_nan' for col in missing],
            ],
        }]

    settings = [] # ...

    # ---  run evaluation  ---
    results = {}
    for o in outliers:
        print("Outliers: ", o)
        start = time()
        result = custom_grid_search(model, params, data, target, cv, scorer, outliers=o, n_jobs=8)
        end = time()
        print("Done in: ", end - start, ' s')
        print(result)
        results[o] = result
    try:
        with open('./results/' + TEST_NAME + "_" + dataset + '_pickle.b', 'wb') as fp:
            pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./results/' + TEST_NAME + "_" + dataset + '_grid.b', 'wb') as fp:
            pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("error")
        traceback.print_exc()
    plot_and_save_results(results, get_class_from_list(settings), './results/' + TEST_NAME + "_" + dataset, precision=precision)

#     start = time()
#     result = custom_grid_search(model, params, data, target, cv, scorer, outliers=0)
#     end = time()
#     print("Done in: ", end - start, ' s')
#     print(result)
#     model_results[dataset] = result
# plot_and_save_results(model_results, get_class, './results/' + TEST_NAME + "_multi", precision=4)

