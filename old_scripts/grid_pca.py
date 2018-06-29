import json
import pickle
import traceback

from sklearn.decomposition import PCA
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
def get_class(params):
    if params['poly']:
        model_name = 'poly ' + str( next(iter(params['poly'].powers_per_column.values())))
    else:
        model_name = 'no poly'
    if params['combinations']:
        label = 'feat. product'
    else:
        label = 'basic'
    print(model_name, label)
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


TEST_NAME = 'dim_red_model_based'
sets = [
    'prudential',
    'boston',
    'houses',
    'heart'
]
MISSING = False
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
        cv = StratifiedKFold(3)
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
        cv = KFold(3, shuffle=True, random_state=0)
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
        cv = KFold(3, shuffle=True, random_state=0)
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
        cv = KFold(3, shuffle=True, random_state=0)
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
        # ('simple_imputer', FillNaTransformer()),
        ('zero_filler', ZeroFiller()),  # just in case there are any left
        # ('main_imputer', HotDeckFullImputer(col_k_pairs={})),
        ('poly', PolynomialsAdder(powers_per_column={col: [2] for col in continuous})),
        ('combinations', FeatureProduct(columns=continuous)),
        ('dropper', FeatureDropper(drop=[])),
        ('boxcox', BoxCoxTransformer(BOX_COX)),
        ('scaler', StandardScaler()),
        ('reduce_dim', None),
        ('predictor', None)
    ])

    one_hot = CustomOneHotEncoder(columns=categorical) if dataset != 'boston' else None
    params = [
        {  # BASELINE
            'onehot': [one_hot],
            'clipper': [None],
            'binner': [None],
            'binner2': [None],
            # 'simple_imputer': [
            #     FillNaTransformer(zero=missing),
            #     FillNaTransformer(mean=missing),
            # ],
            # 'simple_imputer__nan_flag': [missing],
            # 'main_imputer': [
            #     None,
            #     # HotDeckSimpleImputer(columns=missing), # fits to columns without nans
            # ],
            'poly': [
                None,
                # PolynomialsAdder(powers_per_column={col: [2] for col in continuous}),
                # PolynomialsAdder(powers_per_column={col: [3] for col in continuous}),
                PolynomialsAdder(powers_per_column={col: [2, 3] for col in continuous}),
            ],
            'combinations': [
                None,
                # FeatureProduct(columns=continuous[:6]),
                FeatureProduct(columns=continuous)
            ],
            'dropper__drop': [[]], # filter out nan flags
            'boxcox': [None],
            'scaler': [None],
            'reduce_dim': [
                None,
                PCA(50),
                PCA(60),
                PCA(70),
                PCA(80),
                PCA(90),
            ],
            'predictor': predictors
        },
    ]
    settings = [
        ('base', 'None'),
        ('base', 'PCA 50'),
        ('base', 'PCA 60'),
        ('base', 'PCA 70'),
        ('base', 'PCA 80'),
        ('base', 'PCA 90'),
        ('product', 'None'),
        ('product', 'PCA 50'),
        ('product', 'PCA 60'),
        ('product', 'PCA 70'),
        ('product', 'PCA 80'),
        ('product', 'PCA 90'),
        ('^2, ^3', 'None'),
        ('^2, ^3', 'PCA 50'),
        ('^2, ^3', 'PCA 60'),
        ('^2, ^3', 'PCA 70'),
        ('^2, ^3', 'PCA 80'),
        ('^2, ^3', 'PCA 90'),
        ('^2, ^3, product', 'None'),
        ('^2, ^3, product', 'PCA 50'),
        ('^2, ^3, product', 'PCA 60'),
        ('^2, ^3, product', 'PCA 70'),
        ('^2, ^3, product', 'PCA 80'),
        ('^2, ^3, product', 'PCA 90'),
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
    try:
        with open('./results/' + TEST_NAME + "_" + dataset + '_pickle.b', 'wb') as fp:
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

