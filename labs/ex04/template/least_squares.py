# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
import costs


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w_star = np.linalg.solve(np.transpose(tx) @ tx, np.transpose(tx) @ y)
    MSE = costs.compute_mse(y,tx,w_star)
    return w_star, MSE