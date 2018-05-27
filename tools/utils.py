import numpy as np

# apply one or two parameter box_cox
def box_cox(X, lambda1=0, lambda2=0):
    if lambda1:
        return (X + lambda2)**lambda1 / lambda1
    else:
        return np.log1p(X + lambda2)