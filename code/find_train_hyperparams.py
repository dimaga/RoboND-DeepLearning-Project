"""Searches for optimal hyper-parameter values using Bayesian Optimization
method.

The module expects the presence of skopt library, which can be installed
using the following command:

pip install scikit-optimize

See more details at https://github.com/scikit-optimize/scikit-optimize 
"""

import numpy as np
from skopt import gp_minimize

def f(x):
    if x[1] != 2:
        return abs(x[0])

    return -1


res = gp_minimize(f, [(-10.0, 10.0), [1, 2, 3, 4, 5]])

print(res['x'])