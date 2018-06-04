"""
Python implementation of the multi-objective problem given in
@Article{picheny2016bayesian,
  Title                    = {A Bayesian optimization approach to find Nash equilibria},
  Author                   = {Picheny, Victor and Binois, Mickael and Habbal, Abderrahmane},
  Journal                  = {arXiv preprint arXiv:1611.02440},
  Year                     = {2016}
}

"""
import numpy as np
from ne.minimizers.cmaes import CMAES


class MOP1(object):
    def __init__(self, is_log=True, is_minimize=True, is_noise=False):
        self._num_evals = 0
        self._xs = []
        self._fs = []
        self._is_log = is_log
        self._lb = np.array([-5, 0])
        self._ub = np.array([10, 15])
        self._is_minimize = is_minimize
        self._x_ne = np.array([-3.786, 15])
        self._dim = 2
        self._is_noise = is_noise
        self._mc_samples = 1

    def obj_1(self, x):
        """
        x is assumed to be in [lb,ub]
        :param x:
        :return:
        """
        val = (x[1] - 5.1 * (x[0] / (2. * np.pi)) ** 2 + (5. / np.pi) * x[0] - 6.) ** 2 + 10 * (
            (1 - (1. / (8. * np.pi))) * np.cos(x[0]) + 1.)
        return val + 7.5 * self._is_noise * np.random.randn()  # noise is set 0.1 * of the range of the function = 3

    def obj_2(self, x):
        """
         x is assumed to be in [lb, ub]
        :param x:
        :return:
        """
        val = - np.sqrt((10.5 - x[0]) * (x[0] + 5.5) * (x[1] + 0.5)) - (x[1] - 5.1 * (
            x[0] / (2 * np.pi)) ** 2 - 6) ** 2 / 30. - ((1 - 1. / (8 * np.pi)) * np.cos(x[0]) + 1) / 3.
        return val + 3 * self._is_noise * np.random.randn()  # noise is set 0.1 * of the range of the function = 3

    def unit2space(self, x):
        x_space = self._lb + (self._ub - self._lb) * x
        return x_space

    def space2unit(self, x):
        """
        Transforms x from the actual space of the problem to unit hypercube space
        :return:
        """
        x_unit = (x - self._lb) / (self._ub - self._lb)
        return x_unit

    def evaluate(self, _x, is_minimize=None):
        # handy flag for switching minimization into maximization
        # can be used to override the default setup
        is_minimize = self._is_minimize if is_minimize is None else is_minimize
        x = self.unit2space(_x)
        self._num_evals += 1
        val1 = 0
        val2 = 0
        for _ in range(self._mc_samples):
            val1 += self.obj_1(x)
            val2 += self.obj_2(x)
        fs_val = [val1 / self._mc_samples, val2 / self._mc_samples]
        if self._is_log:
            self._xs.append(_x)
            self._fs.append(fs_val)
        if is_minimize:
            return fs_val
        else:
            return [-f_val for f_val in fs_val]

    def get_num_evals(self):
        return self._num_evals

    def get_xs(self):
        res = list(self._xs)
        self._xs = []
        return res

    def get_fs(self):
        res = list(self._fs)
        self._fs = []
        return res

    def get_f_ne(self):
        return self.evaluate(self.space2unit(self._x_ne), is_minimize=True)

    def get_x_ne(self, is_unit=True):
        return self.space2unit(self._x_ne.copy()) if is_unit else self._x_ne.copy()

    def set_minimize(self, is_minimize):
        """

        :param is_minimize:
        :return:
        """
        self._is_minimize = is_minimize

    def ne_regret(self, x0, is_unit=False):
        """
            Measures the unilateral deviation regret given x0
        """
        if self._is_noise:
            self._mc_samples = 10

        if not is_unit:
            x0 = self.space2unit(x0)

        f_val = self.evaluate(x0, is_minimize=True)

        def f_y(y):
            return self.evaluate(np.concatenate((x0[:self._dim // 2], y)), is_minimize=True)[1] - f_val[1]

        def f_x(x):
            return self.evaluate(np.concatenate((x, x0[self._dim // 2:])), is_minimize=True)[0] - f_val[0]



        res = CMAES(f_y, self._dim // 2, x0=x0[:(self._dim // 2)], is_restart=True, max_fevals=1e2).run()
        dev_y = -res[1]

        res = CMAES(f_x, self._dim // 2, x0=x0[(self._dim // 2):], is_restart=True, max_fevals=1e2).run()
        dev_x = -res[1]

        # set mc samples
        self._mc_samples = 1

        return max(dev_x, dev_y)
