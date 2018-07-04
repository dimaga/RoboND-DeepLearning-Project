"""Searches for optimal hyper-parameter values using Bayesian Optimization
method.

The module expects the presence of skopt library, which can be installed
using the following command:

pip install scikit-optimize

See more details at https://github.com/scikit-optimize/scikit-optimize 
"""

import numpy as np
from skopt import gp_minimize, load, dump
import os.path


def f(x):
    if x[1] != 2:
        return abs(x[0])

    return -1


def checkpoint_saver(res):
    dump(res, "./checkpoint.pkl", store_objective=False)


try:
    prev_res = load("./checkpoint.pkl")
except FileNotFoundError:
    prev_res = None

res = gp_minimize(
    f,
    [(-10.0, 10.0), [1, 2, 3, 4, 5]],
    callback=[checkpoint_saver],
    x0=None if prev_res is None else prev_res.x_iters,
    y0=None if prev_res is None else prev_res.func_vals)

print(res['x'])