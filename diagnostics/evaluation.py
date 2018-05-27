import time

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def weighted_quad_kappa(y_test, y_predicted):
    conf_mat = confusion_matrix(y_test, y_predicted, labels=[1,2,3,4,5,6,7,8])
    return conf_mat

def fit_and_eval(model, X, y, k=3):
    kf = KFold(n_splits=3)
    train_times = []
    test_times = []
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index].copy(), X.loc[test_index].copy()
        y_train, y_test = y[train_index], y[test_index]
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        train_times.append(end - start)

        start = time.time()
        y_predicted = model.predict(X_test)
        test_times.append(end - start)

        scores.append(weighted_quad_kappa(y_test, y_predicted))
    return scores, train_times, test_times

