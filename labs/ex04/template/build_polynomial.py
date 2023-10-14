# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    poly = np.zeros(shape = (len(x), degree + 1))
    
    # degree 0 term
    poly[:,0] = 1
    #degree 1 term
    poly[:,1] = x
    #degree higher than 2
    for i in range(2, degree + 1):
        poly[:, i] = np.power(x, i)

    return poly
