from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from diagnostics.evaluation import rev_weighted_quad_kappa
from diagnostics.evaluation import fit_and_eval, rev_weighted_quad_kappa, rmse, error_rate
from imputers.zero_filler import ZeroFiller
from tools.datasets import get_boston, get_heart, get_houses, get_prudential
from diagnostics.grid_search import custom_grid_search
from config import *
from transformers import BoxCoxTransformer, LabelsClipper, OutliersClipper, CustomOneHotEncoder, CustomBinner, CustomBinaryBinner, FeatureDropper
import warnings

from visualization.plot_results import plot_and_save_results

warnings.filterwarnings('ignore')

# get labels and ticks for plotting and report
def get_class(params):
    model_names = []
    if params['scaler'] and isinstance(params['scaler'], StandardScaler):
        model_names.append('standard')
    if params['scaler'] and isinstance(params['scaler'], RobustScaler):
        model_names.append('robust')
    if params['boxcox']:
        model_names.append('boxcox')
    if not model_names:
        model_name =  'base'
    else:
        model_name = ', '.join(model_names)

    if params['clipper']:
        label = 'clipped'
    else:
        label = 'not clipped'
    return model_name, label

TEST_NAME = 'scaling'
sets = [
    'prudential',
    'boston',
    'houses',
    'heart'
]
outliers = [
    0,
#     'clip',
#     'remove'
]



model_results = {}
for dataset in sets:
    print("Performing tests for ", dataset)


    if dataset == 'prudential':
        data, labels, continuous, discrete, dummy, categorical, target = get_prudential()
        train = data.drop(target, axis=1)
        cv = StratifiedKFold(3)
        scorer = rev_weighted_quad_kappa
        predictors = [LabelsClipper(regressor=LinearRegression())]
        # binner = CustomBinaryBinner({ col: {'bins': 7} for col in continuous })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous },
        #     { col: {'bins': 5} for col in continuous },
        #     { col: {'bins': 7} for col in continuous },
        #     { col: {'values': [train[col].max()]} for col in continuous }]
        BOX_COX = BOX_COX_P
        precision = 4

    elif dataset == 'boston':
        data, labels, continuous, discrete, dummy, categorical, target = get_boston()
        train = data.drop(target, axis=1)
        cv = KFold(3, shuffle=True, random_state=0)
        scorer = rmse
        predictors = [LinearRegression()]
        # binner = CustomBinner({ col: {'bins': 7} for col in continuous + discrete })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous + discrete },
        #     { col: {'bins': 5} for col in continuous + discrete },
        #     { col: {'bins': 7} for col in continuous + discrete },
        #     { col: {'values': [train[col].max()]} for col in continuous + discrete }]
        BOX_COX = BOX_COX_B
        precision = 3

    elif dataset == 'houses':
        data, labels, continuous, discrete, dummy, categorical, target = get_houses()
        cv = KFold(3, shuffle=True, random_state=0)
        scorer = rmse
        precision = 0
        predictors = [LinearRegression()]
        train = data.drop(target, axis=1)
        # binner = CustomBinaryBinner({ col: {'values': [train[col].max()]} for col in continuous + discrete })
        # BINNER_CONFIG = [{ col: {'bins': 3} for col in continuous + discrete },
        #     # { col: {'bins': 5} for col in continuous + discrete },
        #     { col: {'bins': 7} for col in continuous + discrete },
        #     { col: {'values': [train[col].max()]} for col in continuous + discrete }]
        BOX_COX = BOX_COX_HO

    elif dataset == 'heart':
        data, labels, continuous, discrete, dummy, categorical, target = get_heart()
        train = data.drop(target, axis=1)
        cv = KFold(3, shuffle=True, random_state=0)
        scorer = error_rate
        predictors = [LogisticRegression()]
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
        ('zero_filler', ZeroFiller()),
        ('boxcox', BoxCoxTransformer(BOX_COX)),
        ('scaler', StandardScaler()),
        ('predictor', None)
    ])

    one_hot = CustomOneHotEncoder(columns=categorical) if dataset != 'boston' else None
    params = [
        {  # BASELINE
            'onehot': [one_hot],
            'clipper': [None, OutliersClipper(continuous)],
            'binner': [None],
            'binner2': [None],
            'boxcox': [None, BoxCoxTransformer(BOX_COX)],
            'scaler': [None, StandardScaler(), RobustScaler()],
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
    plot_and_save_results(results, get_class, './results/' + TEST_NAME + "_" + dataset, precision=precision)

#     start = time()
#     result = custom_grid_search(model, params, data, target, cv, scorer, outliers=0)
#     end = time()
#     print("Done in: ", end - start, ' s')
#     print(result)
#     model_results[dataset] = result
# plot_and_save_results(model_results, get_class, './results/' + TEST_NAME + "_multi", precision=4)

