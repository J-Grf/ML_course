# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss_MSE(y, tx, w):
    """Calculate the loss MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    e = y - tx @ w
    return 1/(2*N) * np.sum(np.square(e))
    
def compute_loss_MAE(y, tx, w):
    """Calculate the loss MAE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    e = y - tx @ w
    return 1/N * np.sum(np.abs(e))
