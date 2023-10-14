# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    N = tx.shape[0]
    D = tx.shape[1]
    lambdaM = (lambda_ * 2 * N) * np.identity(D)
    w_star = np.linalg.solve(np.transpose(tx) @ tx + lambdaM, np.transpose(tx) @ y)
    return w_star
