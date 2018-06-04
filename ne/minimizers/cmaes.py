"""
Python implementation of CMA-ES wrapper
"""
import cma
from ne.minimizers.minimizer import Minimizer
import numpy as np


class CMAES(Minimizer):
    def __init__(self, fct, dim, max_fevals=None, x0=None, seed=None, is_restart=True, is_mirror=False, bounds=None):
        """

        :param fct: objective function
        :param dim: problem dimensionality
        :param max_fevals: maximum number of function evaluations (this is a rough budget as CMAES does not
                            conform to it strictly
        :param x0: initial guess
        :param seed: seed for random runs
        :param is_restart:
        :param is_mirror:
        :param bounds:
        """

        super(CMAES, self).__init__(fct, dim, max_fevals=max_fevals, x0=x0, seed=seed)
        self._opts = cma.CMAOptions()
        self._opts.set('tolfun', 1e-11)
        self._opts['tolx'] = 1e-11
        self._opts['verbose'] = -1
        self._opts['verb_disp'] = 0
        self._opts['verb_log'] = 0
        if max_fevals is not None:
            self._opts['maxfevals'] = max_fevals
        if seed is not None:
            self._opts['seed'] = seed

        self._dim_cma = dim
        if dim == 1:
            self._dim_cma = 2
            self._x0 = np.hstack((self._x0, self._x0))

        if bounds is None:
            self._opts['bounds'] = ([0] * self._dim_cma, [1] * self._dim_cma)
        else:
            self._opts['bounds'] = bounds

        self._is_restart = is_restart
        self._is_mirror = is_mirror

        if self._is_mirror:
            self._opts['CMA_mirrors'] = 1
        else:
            self._opts['CMA_mirrors'] = 0

    def run(self):
        if self._is_restart:
            x0 = '{lb} + ({delta})*(np.random.random({d}))'.format(lb=list(self._opts['bounds'][0]),
                                                                   delta=list(np.array(self._opts['bounds'][1]) -
                                                                              self._opts['bounds'][0]),
                                                                   d=self._dim_cma)
            res = cma.fmin(lambda x: self._fct(x[:self._dim]), x0, 0.25, self._opts, eval_initial_x=True, restarts=5,
                           bipop=True)
        else:
            res = cma.fmin(lambda x: self._fct(x[:self._dim]), self._x0, 0.25, self._opts, eval_initial_x=True,
                           restarts=0)
        x_opt = res[0][:self._dim]  # index trick as 1D is not supported
        f_opt = res[1]
        num_fevals = res[3]  # num of fevals used
        return x_opt, f_opt, num_fevals


if __name__ == "__main__":
    def f(x): return np.sum(10 * (x - 0.5) ** 2)


    Ces = CMAES(f, 2, max_fevals=100, is_restart=False, is_mirror=False, seed=None)
    print (Ces.run())
