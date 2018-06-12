import matplotlib.pyplot as plt
import numpy as np

# up to 3 grid dimensions (choose max from the last one)
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid, cross_val_score



def plot_grid(grid, k_options, l_options, m_options_length, k_label="k options"):
    mean_scores = np.array(-grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(m_options_length, -1, len(k_options))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(k_options)) *
                   (len(l_options) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, l_scores) in enumerate(zip(l_options, mean_scores)):
        plt.bar(bar_offsets + i, l_scores, label=label, color=COLORS[i])

    plt.title("Search for best hyperparams")
    plt.xlabel(k_label)
    plt.xticks(bar_offsets + len(l_options) / 2, k_options)
    plt.ylabel('Mean score')
    plt.legend(loc='best')
    plt.show()


# for g in ParameterGrid(grid):
#     rf.set_params(**g)
#     rf.fit(X,y)
#     # save if best
#     if rf.oob_score_ > best_score:
#         best_score = rf.oob_score_
#         best_grid = g
#
# print "OOB: %0.5f" % best_score
# print "Grid:", best_grid

def eval_single(model, data, target, X_test, y_test, scorer, outliers=None):
    if outliers == 'clip':
        labels_mean = data[target].mean()
        labels_std = data[target].std()
        X_train = data.drop(target, axis=1)
        y_train = data[target].clip(labels_mean - 3 * labels_std, labels_mean + 3 * labels_std)
    elif outliers == 'remove':
        labels_mean = data[target].mean()
        labels_std = data[target].std()
        data_filtered = data[(data[target] - labels_mean).abs() <  3 * labels_std]
        X_train = data_filtered.drop(target, axis=1)
        y_train = data_filtered[target]
    else:
        X_train = data.drop(target, axis=1)
        y_train = data[target]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return scorer(y_test, y_pred)




def evaluate_params(model, params, data, target, cv, scorer, outliers=None):
    scores = []
    model.set_params(**params)
    for train_index, test_index in cv.split(data, data[target]):
        data_test = data.loc[test_index].copy()
        data_train = data.loc[train_index].copy()
        X_test = data_test.drop(target, axis=1)
        y_test = data_test[target]
        scores.append(eval_single(clone(model), data_train, target, X_test, y_test, scorer, outliers))
    scores = np.array(scores)
    return {
        'score': scores.mean(),
        'std': scores.std(),
        'params': params
    }


def custom_grid_search(model, grid, data, target, cv, scorer, outliers=None):
    return Parallel(n_jobs=7)(
        delayed(evaluate_params)(
            clone(model),
            params,
            data.copy(),
            target,
            cv,
            scorer,
            outliers
        ) for params in ParameterGrid(grid)
    )