"""
Python interface for solvers / minimizers
"""
import numpy as np


class Minimizer(object):
    """
    An interface for methods minimizing black-box functions `fct` over the unit hypercube [0,1]^n
    given `max_fevals` function evaluations
    """
    def __init__(self, fct, dim, max_fevals=100, x0=None, seed=None, **kwargs):
        """

        :param fct: objective function
        :param dim: problem dimensionality
        :param max_fevals: number of maximum function evaluations
        :param x0: initial guess
        :param kwargs: options for subclasses
        """
        self._fct = fct
        self._max_fevals = max_fevals
        self._dim = dim
        self._seed = seed

        if seed is not None:
            np.random.seed(seed)

        if x0 is None:
            self._x0 = np.random.random(dim)
        else:
            self._x0 = x0

    def run(self):
        raise NotImplementedError("Inheriting classes should implement this method")
