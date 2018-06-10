import pickle

import numpy as np

# apply one or two parameter box_cox
def box_cox(X, lambda1=0, lambda2=0):
    if lambda1:
        return (X + lambda2)**lambda1 / lambda1
    else:
        return np.log1p(X + lambda2)

REPORT_PATH = './results/'

def save_report(name, report):
    with open("{}{}.pickle".format(REPORT_PATH, name), 'wb') as handle:
        pickle.dump(report, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_report(name):
    with open("{}{}.pickle".format(REPORT_PATH, name), 'rb') as handle:
        return pickle.load(handle)