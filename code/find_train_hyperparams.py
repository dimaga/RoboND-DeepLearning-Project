"""Searches for optimal hyper-parameter values using Bayesian Optimization
method.

The module expects the presence of skopt library, which can be installed
using the following command:

pip install scikit-optimize

See more details at https://github.com/scikit-optimize/scikit-optimize 
"""

import pickle
import glob
from skopt import gp_minimize

MAX_CALLS = 100

def f(x):
    if x[1] != 2:
        return abs(x[0])

    return -1


class CheckpointSaver():
    def __init__(self):
        self.__iteration = 0


    def do(self, res):
        """Saves intermediate parameters of hyper-parameter
        optimization in case the script fails"""

        if len(res.x_iters) >= MAX_CALLS - 10:
            return

        with open("checkpoint{:08}.pkl".format(self.__iteration), "wb") as f:
            p = pickle.Pickler(f)
            p.dump(res.x_iters)
            p.dump(res.func_vals)

        self.__iteration += 1


def checkpoint_loader(fileName):
    """Loads hyper-parameter optimization parameters to start
    learning from previously saved checkpoint"""

    with open(fileName, "rb") as f:
        u = pickle.Unpickler(f)
        x_iters = u.load()
        func_vels = u.load()

    return x_iters, func_vels


checkpoints = sorted(glob.glob("*.pkl"))
if len(checkpoints) > 1:
    # The last checkpoint may be broken if the process was interrupted while
    # saving it, so load the checkpoint before the last one
    x0, y0 = checkpoint_loader(checkpoints[-2])
else:
    x0 = None
    y0 = None

n_calls = MAX_CALLS
if x0 is not None:
    n_calls += -len(x0)

res = gp_minimize(
    f,
    [(-10.0, 10.0), [1, 2, 3, 4, 5]],
    callback=[CheckpointSaver().do],
    x0=x0,
    y0=y0,
    n_calls=n_calls)

print(res['x'])
