import time
import numpy as np
from sklearn import clone
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

def histogram(y):
    hist = np.array([0] * 8)
    for r in y:
        hist[r-1] += 1
    return hist

def rmse(y_test, y_predicted, **kwargs):
    return np.sqrt(mean_squared_error(y_test, y_predicted))

def error_rate(y_test, y_predicted, **kwargs):
    return 1 - accuracy_score(y_test, y_predicted)

def rev_weighted_quad_kappa(y_test, y_predicted, **kwargs):
    conf_mat = confusion_matrix(y_test, y_predicted, labels=[1,2,3,4,5,6,7,8])
    N = 8
    total = len(y_test)

    numerator = 0.0
    denominator = 0.0
    hist_p = histogram(y_predicted)
    hist_t = histogram(y_test)
    for i in range(N):
        for j in range(N):
            expected_count = hist_p[i] * hist_t[j] / float(total)
            w = pow(i - j, 2.0) / pow(N - 1, 2.0)
            numerator += w * conf_mat[i][j]
            denominator += w * expected_count

    return numerator / denominator

# evaluates model on given data
def fit_and_eval_single(model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start

    start = time.time()
    y_predicted = model.predict(X_test)
    end = time.time()
    test_time = end - start

    score = rev_weighted_quad_kappa(y_test, y_predicted)
    return score, train_time, test_time

# evaluates model using k-fold cv in parallel
def fit_and_eval(model, X, y, k=3):
    kf = KFold(n_splits=k)
    return Parallel(n_jobs=8)(delayed(fit_and_eval_single)(
        clone(model),
        X.iloc[train_index].copy(), X.iloc[test_index].copy(),
        y[train_index].copy(), y[test_index].copy()
    ) for train_index, test_index in kf.split(X))