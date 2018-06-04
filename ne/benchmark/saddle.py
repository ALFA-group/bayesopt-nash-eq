import numpy as np
from ne.minimizers.cmaes import CMAES


class Saddle(object):
    def __init__(self, is_log=True, is_minimize=True, is_noise=False, dim=2, x_opt=0.5):
        self._num_evals = 0
        self._xs = []
        self._fs = []
        self._is_log = is_log
        self._is_minimize = is_minimize
        self._is_noise = is_noise
        self._x_ne = np.array([x_opt] * dim)
        # self._x_ne = np.array([0.25, 0.75])
        self._dim = dim
        self._mc_samples = 1

    def obj_1(self, x):
        """
        x is assumed to be in [lb,ub]
        :param x:
        :return:
        """
        n = self._dim // 2 - 1
        val = 0
        for i, _x in enumerate(x):
            if i > n:
                val = val - (_x - self._x_ne[i]) ** 2
            else:
                val = val + (_x - self._x_ne[i]) ** 2
        return val + 0.025 * self._is_noise * np.random.randn()

    def evaluate(self, _x, is_minimize=None):
        # handy flag for switching minimization into maximization
        # can be used to override the default setup
        is_minimize = self._is_minimize if is_minimize is None else is_minimize
        self._num_evals += 1
        val = 0
        for _ in range(self._mc_samples):
            val += self.obj_1(_x)
        val = val / self._mc_samples
        fs_val = [val, -val]
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
        return self.evaluate(self._x_ne, is_minimize=True)

    def get_x_ne(self):
        """
        for compliance with mop1 purpose is_unit is used here
        :return:
        """
        return self._x_ne.copy()

    def ne_regret(self, x0):
        """
            Measures the unilateral deviation regret given x0
        """

        if self._is_noise:
            self._mc_samples = 10

        f_val = self.evaluate(x0, is_minimize=True)

        def f_y(y):
            return self.evaluate(np.concatenate((x0[:self._dim // 2], y)), is_minimize=True)[1] - f_val[1]

        def f_x(x):
            return self.evaluate(np.concatenate((x, x0[self._dim // 2:])), is_minimize=True)[0] - f_val[0]


        res = CMAES(f_y, self._dim // 2, x0=x0[:(self._dim // 2)], is_restart=True, max_fevals=50).run()
        dev_y = -res[1]



        res = CMAES(f_x, self._dim // 2, x0=x0[(self._dim // 2):], is_restart=True,
                    max_fevals=50).run()  # 1e1 * self._dim // 2
        dev_x = -res[1]

        # reset the number of mc samples
        self._mc_samples = 1

        return max(dev_x, dev_y)
