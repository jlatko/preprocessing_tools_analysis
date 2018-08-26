import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.model_selection import learning_curve
from tqdm import tqdm


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.2, 1.0, 5),
                        scoring="neg_mean_squared_error"):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_triple_curve(estimator, title, data, target, X_test, y_test, cv, scoring, outliers, ylim=None,
                        n_jobs=1, train_sizes=np.linspace(.2, 1.0, 5)):
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(11,8))
    # plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("relative training set size")
    plt.ylabel("error")
    train_sizes, train_scores, cv_scores, test_scores = triple_curve(
        estimator, data, target, X_test, y_test, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=0)
    train_scores_std = np.std(train_scores, axis=0)
    cv_scores_mean = np.mean(cv_scores, axis=0)
    cv_scores_std = np.std(cv_scores, axis=0)
    test_scores_mean = np.mean(test_scores, axis=0)
    test_scores_std = np.std(test_scores, axis=0)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, cv_scores_mean - cv_scores_std,
                     cv_scores_mean + cv_scores_std, alpha=0.1,
                     color="g")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="train")
    plt.plot(train_sizes, cv_scores_mean, 'o-', color="g",
             label="cv")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="test")

    plt.legend(loc="best")

    # plt.yscale('log')
    plt.tight_layout()
    return plt


def triple_curve(model, data, target, X_test, y_test, cv, train_sizes, scoring, n_jobs=1, outliers=None):
    test_scores = [[] for i in range(cv.n_splits)]
    train_scores = [[] for i in range(cv.n_splits)]
    cv_scores = [[] for i in range(cv.n_splits)]
    for i, (train_index, test_index) in enumerate(cv.split(data, data[target])):
        size = len(train_index)
        train_chunks_sizes = [int(size * chunk) for chunk in train_sizes]
        data_test = data.loc[test_index].copy()
        data_train = data.loc[train_index].copy()
        X_cv = data_test.drop(target, axis=1)
        y_cv = data_test[target]
        print(i)
        for chunk_size in tqdm(train_chunks_sizes):
            data_tmp = data_train.iloc[:chunk_size]
            X_tmp = data_tmp.drop(target, axis=1)
            y_tmp = data_tmp[target]
            model_tmp = clone(model)
            model_tmp.fit(X_tmp, y_tmp)
            print(model_tmp.predict(X_tmp.copy()))
            train_scores[i].append(scoring(y_tmp, model_tmp.predict(X_tmp.copy())))
            cv_scores[i].append(scoring(y_cv, model_tmp.predict(X_cv.copy())))
            test_scores[i].append(scoring(y_test, model_tmp.predict(X_test.copy())))
        print(train_scores[i])
        print(test_scores[i])

    return train_sizes, np.array(train_scores), np.array(cv_scores), np.array(test_scores)