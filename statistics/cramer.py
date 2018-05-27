import pandas as pd
import numpy as np
import math
import scipy.stats as stats

# returns Cramer's V and chi2 p-value
def cramers_v(data, col1, col2):
    cross_tab = pd.crosstab(data[col1], data[col2])

    chi_stats = stats.chi2_contingency(cross_tab)
    n = cross_tab.sum().sum()
    r,k = cross_tab.shape
    v = math.sqrt((chi_stats[0] / n) / min(k - 1, r - 1))
    return v, chi_stats[1]

def categorical_relation_with(data, target, cols):
    cross = np.array([ cramers_v(data, cols[i], target) for i in range(len(cols))])

    v_table = pd.Series(cross[:, 0], index=cols)
    p_table = pd.Series(cross[:, 1], index=cols)
    return v_table, p_table

def cross_categorical(data, cols):
    cross = np.array([
        [cramers_v(data,cols[i], cols[j]) if i < j else (1, 0)
         for j in range(len(cols))]
        for i in range(len(cols))
    ])
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i > j:
                cross[i,j] = cross[j,i]

    v_table = pd.DataFrame(cross[:,:,0], index=cols, columns=cols)
    p_table = pd.DataFrame(cross[:,:,1], index=cols, columns=cols)
    return v_table, p_table
