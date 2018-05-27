import pandas as pd
import numpy as np
import math
import scipy.stats as stats

def apply_across(data, categorical, numerical, metric):
    return np.array([[
        metric(*list(data.groupby(cat)[num].apply(np.array))) for cat in categorical
    ] for num in numerical])


# TODO improve
def spearman_with(data, target, numerical):
    return {
        num: stats.spearmanr(data[target], data[num])[0]
    for num in numerical}


def cross_kruskal(data, categorical, numerical):
    return np.array([[
        stats.kruskal(*list(data.groupby(cat)[num].apply(np.array))) for cat in categorical
    ] for num in numerical])

def cross_anova(data, categorical, numerical):
    return np.array([[
        stats.f_oneway(*list(data.groupby(cat)[num].apply(np.array))) for cat in categorical
    ] for num in numerical])
