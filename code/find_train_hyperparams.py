"""Searches for optimal hyper-parameter values using Bayesian Optimization
method.

The module expects the presence of skopt library, which can be installed
using the following command:

pip install scikit-optimize

See more details at https://github.com/scikit-optimize/scikit-optimize 
"""

import pickle
from skopt import gp_minimize

MAX_CALLS = 100

def f(x):
    if x[1] != 2:
        return abs(x[0])

    return -1


def checkpoint_saver(res):
    """Saves intermediate parameters of hyper-parameter
    optimization in case the script fails"""

    if len(res.x_iters) >= MAX_CALLS - 10:
        return

    with open("./checkpoint.pkl", "wb") as f:
        p = pickle.Pickler(f)
        p.dump(res.x_iters)
        p.dump(res.func_vals)


def checkpoint_loader():
    """Loads hyper-parameter optimization parameters to start
    learning from previously saved state"""

    with open("./checkpoint.pkl", "rb") as f:
        u = pickle.Unpickler(f)
        x_iters = u.load()
        func_vels = u.load()

    return x_iters, func_vels


try:
    x0, y0 = checkpoint_loader()
except FileNotFoundError:
    x0 = None
    y0 = None

n_calls = MAX_CALLS
if x0 is not None:
    n_calls += -len(x0)

res = gp_minimize(
    f,
    [(-10.0, 10.0), [1, 2, 3, 4, 5]],
    callback=[checkpoint_saver],
    x0=x0,
    y0=y0,
    n_calls=n_calls)

print(res['x'])
