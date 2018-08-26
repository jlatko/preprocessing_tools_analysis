import json
import pickle
import traceback

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor

from diagnostics.evaluation import rev_weighted_quad_kappa
from diagnostics.evaluation import fit_and_eval, rev_weighted_quad_kappa, rmse, error_rate
from imputers import *
from tools.datasets import get_boston, get_heart, get_houses, get_prudential
from diagnostics.grid_search import custom_grid_search
from config import *
from transformers import *
import warnings

from visualization.plot_results import plot_and_save_results

warnings.filterwarnings('ignore')

# get labels and ticks for plotting and report
# def get_class(params):
#     model_names = []
#     imputer1 = params['simple_imputer']
#     imputer2 = params['main_imputer']
#     if imputer1:
#         if imputer1.mean:
#             model_names.append('mn')
#         elif imputer1.median:
#             model_names.append('med')
#         else:
#             model_names.append('0')
#     if imputer2:
#         model_names = ['h-deck'] + model_names
#     if params['dropper__drop']:
#         model_names.append('nan')
#     model_name = ' '.join(model_names)
#     if imputer2:
#         label = 'k=' + str(params['main_imputer__default_k'])
#     else:
#         label = 'basic'
#     # label = type(params['predictor']).__name__
#     return model_name, label

# def get_class(params):
#     model_names = []
#     imputer1 = params['simple_imputer']
#     imputer2 = params['main_imputer']
#     if imputer1:
#         if imputer1.mean:
#             model_names.append('mn')
#         elif imputer1.median:
#             model_names.append('med')
#         else:
#             model_names.append('0')
#     if imputer2:
#         model_names = ['h-knn'] + model_names
#     if params['dropper__drop']:
#         model_names.append('nan')
#     model_name = ' '.join(model_names)
#     if imputer2:
#         label = 'k=' + str(params['main_imputer__model__n_neighbors'])
#     else:
#         label = 'basic'
#     # label = type(params['predictor']).__name__
#     return model_name, label

def get_class(params):
    labels = []
    imputer1 = params['simple_imputer']
    imputer2 = params['main_imputer']
    if imputer1:
        if imputer1.mean:
            labels.append('mean')
        elif imputer1.median:
            labels.append('median')
        else:
            labels.append('0')
    if imputer2:
        model_name = type(params['main_imputer'].model).__name__
    else:
        model_name = 'base'
    if params['dropper__drop']:
        labels.append('nan')
    label = ' '.join(labels)
    # if imputer2:
    #     label = 'k=' + str(params['main_imputer__model__n_neighbors'])
    # else:
    #     label = 'basic'
    # label = type(params['predictor']).__name__
    return model_name, label


TEST_NAME = 'imputing_1'
sets = [
    # 'prudential',
    'boston',
    # 'houses',
    'heart'
]
MISSING = True
outliers = [
    0,
#     'clip',
#     'remove'
]



model_results = {}
for dataset in sets:
    print("Performing tests for ", dataset)


    if dataset == 'prudential':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_prudential()
        train = data.drop(target, axis=1)
        cv = StratifiedKFold(5)
        scorer = rev_weighted_quad_kappa
        predictors = [
            LabelsClipper(regressor=LinearRegression()),
            # DecisionTreeClassifier(),
            # KNeighborsClassifier(n_neighbors=5),
        ]
        # binner = CustomBinaryBinner({ col: {'bins': 7} for col in continuous })
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
            LinearRegression(),
            # SVR(),
            # RandomForestRegressor(),
            # DecisionTreeRegressor(),
            # KNeighborsRegressor(n_neighbors=5),
        ]
        # binner = CustomBinner({ col: {'bins': 7} for col in continuous + discrete })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous + discrete },
        #     { col: {'bins': 5} for col in continuous + discrete },
        #     { col: {'bins': 7} for col in continuous + discrete },
        #     { col: {'values': [train[col].max()]} for col in continuous + discrete }]
        BOX_COX = BOX_COX_B
        precision = 3

    elif dataset == 'houses':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_houses()
        cv = KFold(5, shuffle=True, random_state=0)
        scorer = rmse
        precision = 0
        predictors = [
            LinearRegression(),
            # DecisionTreeRegressor(),
            # KNeighborsRegressor(n_neighbors=5),
        ]
        train = data.drop(target, axis=1)
        # binner = CustomBinaryBinner({ col: {'values': [train[col].max()]} for col in continuous + discrete })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous + discrete },
        #     # { col: {'bins': 5} for col in continuous + discrete },
        #     { col: {'bins': 7} for col in continuous + discrete },
        #     { col: {'values': [train[col].max()]} for col in continuous + discrete }]
        BOX_COX = BOX_COX_HO

    elif dataset == 'heart':
        data, labels, continuous, discrete, dummy, categorical, target, missing = get_heart(missing=MISSING)
        train = data.drop(target, axis=1)
        cv = KFold(5, shuffle=True, random_state=0)
        scorer = error_rate
        predictors = [
            LogisticRegression(),
            # SVC(),
            # RandomForestClassifier(),
            # DecisionTreeClassifier(),
            # KNeighborsClassifier(n_neighbors=5),
        ]
        # BINNER_CONFIG = { col: {'bins': 3} for col in continuous + discrete }
        # binner = CustomBinner({ col: {'bins': 3} for col in continuous + discrete })
        BOX_COX = BOX_COX_HE
        precision = 4

    else:
        continue


    model = Pipeline([
        ('onehot', CustomOneHotEncoder(columns=categorical)),
        ('clipper', OutliersClipper(continuous)),
        ('binner', CustomBinaryBinner({})),
        ('binner2', CustomBinaryBinner({})),
        ('simple_imputer', FillNaTransformer()),
        # ('zero_filler', ZeroFiller()),  # just in case there are any left
        ('main_imputer', HotDeckFullImputer(col_k_pairs={})),
        ('dropper', FeatureDropper(drop=[])),
        ('boxcox', BoxCoxTransformer(BOX_COX)),
        ('scaler', StandardScaler()),
        ('predictor', None)
    ])
    one_hot = CustomOneHotEncoder(columns=categorical) if dataset != 'boston' else None
    params = [
        {  # BASELINE
            'onehot': [one_hot],
            'clipper': [None],
            'binner': [None],
            'binner2': [None],
            'simple_imputer': [
                FillNaTransformer(zero=missing),
                FillNaTransformer(mean=missing),
            ],
            # 'simple_imputer__nan_flag': [missing],
            'main_imputer': [
                None,
                # HotDeckSimpleImputer(columns=missing), # fits to columns without nans
            ],
            'dropper__drop': [[]], # filter out nan flags
            'boxcox': [None],
            'scaler': [None],
            'predictor': predictors
        },
        {  # fill
            'onehot': [one_hot],
            'clipper': [None],
            'binner': [None],
            'binner2': [None],
            'simple_imputer': [
                FillNaTransformer(zero=missing),
                FillNaTransformer(mean=missing),
                FillNaTransformer(median=missing),
            ],
            'simple_imputer__nan_flag': [missing],
            'main_imputer': [
                ModelBasedFullImputer(columns=missing, model=DecisionTreeRegressor(max_depth=8)),
                ModelBasedFullImputer(columns=missing, model=XGBRegressor(max_depth=3)),
                ModelBasedFullImputer(columns=missing, model=LinearRegression()),
                ModelBasedFullImputer(columns=missing, model=RandomForestRegressor()),
            ],
            'dropper__drop': [[], [col + "_nan" for col in missing]], # filter out nan flags
            'boxcox': [None],
            'scaler': [None],
            'predictor': predictors
        },
    ]

    results = {}
    for o in outliers:
        print("Outliers: ", o)
        start = time()
        result = custom_grid_search(model, params, data, target, cv, scorer, outliers=o)
        end = time()
        print("Done in: ", end - start, ' s')
        print(result)
        results[o] = result
    # try:
    #     with open('./results/' + TEST_NAME + "_" + dataset + '_pickle.b', 'wb') as fp:
    #         pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # except:
    #     print("error")
    #     traceback.print_exc()
    plot_and_save_results(results, get_class, './results/' + TEST_NAME + "_" + dataset, precision=precision)

#     start = time()
#     result = custom_grid_search(model, params, data, target, cv, scorer, outliers=0)
#     end = time()
#     print("Done in: ", end - start, ' s')
#     print(result)
#     model_results[dataset] = result
# plot_and_save_results(model_results, get_class, './results/' + TEST_NAME + "_multi", precision=4)



    # params = [
    #     {  # BASELINE
    #         'onehot': [one_hot],
    #         'clipper': [None],
    #         'binner': [None],
    #         'binner2': [None],
    #         'simple_imputer': [
    #             FillNaTransformer(zero=missing),
    #             FillNaTransformer(mean=missing),
    #         ],
    #         # 'simple_imputer__nan_flag': [missing],
    #         'main_imputer': [
    #             None,
    #             # HotDeckSimpleImputer(columns=missing), # fits to columns without nans
    #         ],
    #         'dropper__drop': [[]], # filter out nan flags
    #         'boxcox': [None],
    #         'scaler': [None],
    #         'predictor': predictors
    #     },
    #     {  # fill
    #         'onehot': [one_hot],
    #         'clipper': [None],
    #         'binner': [None],
    #         'binner2': [None],
    #         'simple_imputer': [
    #             FillNaTransformer(zero=missing),
    #             FillNaTransformer(mean=missing),
    #         ],
    #         'simple_imputer__nan_flag': [missing],
    #         'main_imputer': [
    #             # HotDeckFullImputer(col_k_pairs=[(col, None) for col in missing]),
    #             ModelBasedFullImputer(columns=missing, model=KNeighborsRegressor())
    #         ],
    #         # 'main_imputer__model': [
    #         #     KNeighborsRegressor()
    #         # ],
    #         'main_imputer__model__n_neighbors': [1, 3, 5, 7, 9],
    #         # 'main_imputer__default_k': [3, 5, 13, 17],
    #         'dropper__drop': [[], [col + "_nan" for col in missing]], # filter out nan flags
    #         'boxcox': [None],
    #         'scaler': [None],
    #         'predictor': predictors
    #     },
    # ]