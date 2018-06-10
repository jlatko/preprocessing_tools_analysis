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

def fit_and_eval(model, X, y, k=3):
    kf = KFold(n_splits=k)
    return Parallel(n_jobs=8)(delayed(fit_and_eval_single)(
        clone(model),
        X.iloc[train_index].copy(), X.iloc[test_index].copy(),
        y[train_index].copy(), y[test_index].copy()
    ) for train_index, test_index in kf.split(X))

#
#
# def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
#     """
#     Returns the confusion matrix between rater's ratings
#     """
#     assert(len(rater_a) == len(rater_b))
#     if min_rating is None:
#         min_rating = min(rater_a + rater_b)
#     if max_rating is None:
#         max_rating = max(rater_a + rater_b)
#     num_ratings = int(max_rating - min_rating + 1)
#     conf_mat = [[0 for i in range(num_ratings)]
#                 for j in range(num_ratings)]
#     for a, b in zip(rater_a, rater_b):
#         conf_mat[a - min_rating][b - min_rating] += 1
#     return conf_mat
#
#
# def histogram(ratings, min_rating=None, max_rating=None):
#     """
#     Returns the counts of each type of rating that a rater made
#     """
#     if min_rating is None:
#         min_rating = min(ratings)
#     if max_rating is None:
#         max_rating = max(ratings)
#     num_ratings = int(max_rating - min_rating + 1)
#     hist_ratings = [0 for x in range(num_ratings)]
#     for r in ratings:
#         hist_ratings[r - min_rating] += 1
#     return hist_ratings
#
#
# def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
#     """
#     Calculates the quadratic weighted kappa
#     quadratic_weighted_kappa calculates the quadratic weighted kappa
#     value, which is a measure of inter-rater agreement between two raters
#     that provide discrete numeric ratings.  Potential values range from -1
#     (representing complete disagreement) to 1 (representing complete
#     agreement).  A kappa value of 0 is expected if all agreement is due to
#     chance.
#     quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
#     each correspond to a list of integer ratings.  These lists must have the
#     same length.
#     The ratings should be integers, and it is assumed that they contain
#     the complete range of possible ratings.
#     quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
#     is the minimum possible rating, and max_rating is the maximum possible
#     rating
#     """
#     rater_a = np.array(rater_a, dtype=int)
#     rater_b = np.array(rater_b, dtype=int)
#     assert(len(rater_a) == len(rater_b))
#     if min_rating is None:
#         min_rating = min(min(rater_a), min(rater_b))
#     if max_rating is None:
#         max_rating = max(max(rater_a), max(rater_b))
#     conf_mat = confusion_matrix(rater_a, rater_b,
#                                 min_rating, max_rating)
#     num_ratings = len(conf_mat)
#     num_scored_items = float(len(rater_a))
#
#     hist_rater_a = histogram(rater_a, min_rating, max_rating)
#     hist_rater_b = histogram(rater_b, min_rating, max_rating)
#
#     numerator = 0.0
#     denominator = 0.0
#
#     for i in range(num_ratings):
#         for j in range(num_ratings):
#             expected_count = (hist_rater_a[i] * hist_rater_b[j]
#                               / num_scored_items)
#             d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
#             numerator += d * conf_mat[i][j] / num_scored_items
#             denominator += d * expected_count / num_scored_items
#
#     return 1.0 - numerator / denominator