"""
Python implementation for bayesian approach to compute approximate NEQ
"""
from __future__ import division

from math import erf
import numpy as np
import random
import warnings

from pyDOE import lhs
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W

from ne.minimizers.cmaes import CMAES
from ne.utils.plots import plot_marginalized_responses


class BayesNE(object):
    def __init__(self, fct, n_xs, max_fevals=20, seed=3, is_approx=True, crit='abr', epsilon=0.1):
        """
        :param fct: vectorial payoff function f:X^{sum(n_xs)} --> R^{num_players}, these are to be maximized
        :param n_xs: number of variables for each player
                    e.g., player 1 has n_xs[0] decision variables
                    It is assumed that all the decision var are in [0,1]
        :param max_fevals: function evaluations budget (# oracle calls)
        :param seed: random seed
        :param crit: acquistion function which could be either 
        			'abr':
        			'br':
        			'std': -1 * highest posterior std among all players
        :param is_approx: to compute the estimated regret exactly or analytically

        """
        np.random.seed(seed)
        random.seed(seed)
        self._crit = crit
        self._epsilon = epsilon
        self._is_approx = is_approx
        self.n_xs = n_xs
        self.fct = fct
        self.num_players = len(self.n_xs)
        self.dims = sum(self.n_xs)

        self.errs = []
        self.models = []
        for _ in range(self.num_players):
            kernel = C(1.0, (1e-3, 1e3)) * RBF(np.ones(self.dims), (1e-2, 1e2)) + W()
            self.models.append(GPR(kernel=kernel, n_restarts_optimizer=2))

        self.pts = None
        self.fs_pts = [[] for _ in range(self.num_players)]

        # initial design
        self._max_fevals = int(0.75 * max_fevals)
        self.pts = lhs(self.dims, max_fevals - self._max_fevals)
        for pt in self.pts:
            f_vals = self.fct(pt)
            for p in range(self.num_players):
                self.fs_pts[p].append(f_vals[p])
        for p in range(self.num_players):
            self.models[p].fit(self.pts, self.fs_pts[p])

    def _get_player_indices(self, p):
        _offset = int(np.sum(self.n_xs[:p]))
        return [_ for _ in range(_offset, _offset + self.n_xs[p])]

    def _get_other_players_indices(self, p):
        return [_ for _ in range(self.dims) if _ not in self._get_player_indices(p)]

    def predict(self, xs, p, return_std=True):
        """
            predict the payoff value for player p given joint actions xs
        :param return_std:
        :param xs: players joint actions
        :param p: the player number whose payoff we would like to predict
        :return:
        """
        mu, std = self.models[p].predict(np.atleast_2d(xs), return_std=True)
        if return_std:
            return mu, std
        else:
            return mu

    def predict_best_response(self, xs0, p):
        """
            Predicts best response for player p given the joint actions of other player from xs0
        :param xs0: joint actions of all the players including p
        :param p: player number
        :return: the best response to xs0[-p] and the deviation in p's payoff
        """
        mu0, _ = self.predict(xs0, p)

        # best response minimizes the following fct
        def f(x):
            # xs
            xs = xs0.copy()
            xs[self._get_player_indices(p)] = x
            mu, _ = self.predict(xs, p)
            return np.asscalar((mu0 - mu))

        res_x = CMAES(f, self.n_xs[p], max_fevals=125 * self.dims,
                      x0=xs0.copy()[self._get_player_indices(p)], is_restart=True
                      ).run()

        # return best response and its gain (similarly gain also denotes xs0[p]'s regret)
        return res_x[0], - res_x[1]

    def predict_best_responses(self, xs0, return_maximum_gain=True):
        """
            an iterator over `predict_best_response` for all players
        :param xs0: joint actions of all players
        :param return_maximum_gain: flag to replace maximum gain (which denotes xs0'regret)
        :return:
        """
        best_responses = []
        maximum_gain = None
        for p in range(self.num_players):
            best_response, best_gain = self.predict_best_response(xs0, p)
            if return_maximum_gain:
                maximum_gain = max(best_gain, maximum_gain)
            else:
                best_responses.append(best_response)

        return maximum_gain if return_maximum_gain else best_responses

    def predict_expected_response(self, xs0, p, return_std=True, over_p=True):
        """
            computes the expected response for a player p given its actions, which
            are part of xs0
        :param return_std:
        :param over_p: True denotes expectationa as fct of {p} otherwise it is {-p}
        :param xs0: present joint actions of all players
        :param p: player number
        :return_std: return std of the expected response as well as the second moment of the integral
        :is_exact: computes the exact expected response or empirical
        :return:
        """
        if self._is_approx:
            res = self._approx_expected_response(xs0, p, return_std=return_std, over_p=over_p)
        else:
            res = self._exact_expected_response(xs0, p, return_std=return_std, over_p=over_p)
        return res

    def _approx_expected_response(self, xs0, p, return_std=True, over_p=True):
        """
            approximate the expected response for a player p given xs0 either
            over its actions or other players' actions
        :param return_std:
        :param over_p: True denotes expectationa as fct of {p} otherwise it is {-p}
        :param xs0: present joint actions of all players
        :param p: player number
        :return_std: return std of the expected response as well as the second moment of the integral
        :return:
        """
        # indices for computing expected response w.r.t. p or -p
        l_indices = self._get_other_players_indices(p) if over_p else self._get_player_indices(p)
        # samples
        x_samples = lhs(len(l_indices), 10 * len(l_indices))

        def evaluate_xsample(x_sample):
            _x = xs0.copy()
            _x[l_indices] = x_sample
            return self.predict(_x, p, return_std=False)
        f_samples = np.apply_along_axis(evaluate_xsample, 1, x_samples)

        if return_std:
            return np.mean(f_samples), np.std(f_samples)
        else:
            return np.mean(f_samples)

    def _exact_expected_response(self, xs0, p, return_std=True, over_p=True, compute_mu_std=False):
        """
            exactly compute the expected response for a player p given xs0 either
            over its actions or other players' actions
        :param return_std:
        :param over_p: True denotes expectationa as fct of {p} otherwise it is {-p}
        :param xs0: present joint actions of all players
        :param p: player number
        :return_std: return std of the expected response as well as the second moment of the integral
        :return:
        """
        # indices for computing expected response w.r.t. p or -p
        j_indices = self._get_player_indices(p) if over_p else self._get_other_players_indices(p)
        l_indices = self._get_other_players_indices(p) if over_p else self._get_player_indices(p)
        # get values for evaulating the equation from the paper
        x_p = xs0[j_indices].copy()
        L = self.models[p].L_
        k = self.models[p].kernel_
        fs = np.array(self.fs_pts[p])
        pts = list(self.pts)

        v = k.get_params()['k2__noise_level']
        c = k.get_params()['k1__k1__constant_value']
        ds = 1. / k.get_params()['k1__k2__length_scale'] ** 2

        # computing qs
        tol = 1e-6
        qs = np.zeros(len(pts))
        for j in range(len(pts)):
            pt = pts[j]
            dist_xj = (x_p - pt[j_indices])
            qs[j] = c * np.exp(-0.5 * np.dot(dist_xj, dist_xj * ds[j_indices]))
            for l in l_indices:
                di2 = np.sqrt(ds[l] / 2.)
                qs[j] *= np.sqrt(0.5 * np.pi / ds[l]) * (erf(pt[l] * di2) - erf((pt[l] - 1) * di2))
            qs[j] += v * np.all(np.abs(pt[j_indices] - x_p) < tol)

        mu = np.dot(np.linalg.solve(L, qs), np.linalg.solve(L, fs))

        if return_std:
            # std of mu as a GP
            mu_std = 0
            if compute_mu_std:
                mu_std = c
                for l in l_indices:
                    mu_std *= (np.sqrt(2 * np.pi * ds[l]) * erf(np.sqrt(ds[l] / 2.)) + 2. * np.exp(- ds[l] / 2.) - 2) / ds[l]
                mu_std += v - np.dot(np.linalg.solve(L, qs), np.linalg.solve(L, qs))
                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                if mu_std < 0:
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to noise variance.")
                    mu_std = v

            # std of the responses
            Q = np.zeros((len(pts),) * 2)
            for _p in range(len(pts)):
                pt_p = pts[_p]
                dist_p = x_p - pt_p[j_indices]
                ind_p = np.all(np.abs(dist_p) < tol)
                for _q in range(_p, len(pts)):
                    pt_q = pts[_q]
                    dist_q = x_p - pt_q[j_indices]
                    ind_q = np.all(np.abs(dist_q) < tol)
                    dist_pq = pt_p - pt_q
                    Q[_p, _q] = v**2 * ind_p * ind_q * (np.all(np.abs(dist_pq) < tol))
                    Q[_p, _q] += v * c * (ind_p + ind_q) * np.exp(-0.5 * np.dot(dist_pq, ds * dist_pq))
                    prod_term = c**2 * np.exp(-0.5 * np.dot(dist_p, ds[j_indices] * dist_p)) * \
                                np.exp(-0.5 * np.dot(dist_q, ds[j_indices] * dist_q))
                    for l in l_indices:
                        prod_term *= 0.5 * np.sqrt(np.pi /  ds[l]) * np.exp(-0.25 * ds[l] * (pt_p[l] - pt_q[l])**2)  * (
                            erf(0.5 * np.sqrt(ds[l]) * (pt_p[l] + pt_q[l])) - erf(0.5 * np.sqrt(ds[l]) * (pt_p[l] + pt_q[l] - 2))
                        )

                    Q[_p, _q] += prod_term
                    Q[_q, _p] = Q[_p, _q]

            right_term = np.linalg.solve(L.T, np.linalg.solve(L, fs))
            std =  np.dot(right_term.T, np.dot(Q + np.finfo(np.float32).eps * np.eye(len(pts)), right_term)) - mu**2
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            if std < 0:
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to noise variance.")
                std = v

            if compute_mu_std:
                return mu, np.sqrt(std), np.sqrt(mu_std)
            else:
                return mu, np.sqrt(std)
        else:
            return mu

    def approx_regret(self, xs0, p):
        mu0, std0 = self.predict(xs0, p)
        mu, std = self.predict_expected_response(xs0, p, over_p=False)
        return np.asscalar((mu + 2.32635 * std - mu0) / (std + np.finfo(np.float32).eps))

    def approx_max_regret(self, xs0, is_max=True):
        regret = -np.inf if is_max else 0
        for p in range(self.num_players):
            if is_max:
                regret = max(regret, self.approx_regret(xs0, p))
            else:
                regret += self.approx_regret(xs0, p)
        return regret

    def std(self, xs0, p):
        _, std = self.predict(xs0, p, return_std=True)
        return np.asscalar(std)

    def max_std(self, xs0):
        val = -np.inf
        for p in range(self.num_players):
            # sinc we like to find sol with max std
            val = max(val, self.std(xs0, p))
        return val

    def acq_fct(self, xs):
        """
           A test for xs, the lower acq_fct the more likely xs is a good solution
        :param xs:
        :return: scalar value (float) denoting the penalty of choosing xs
        """
        if self._crit == 'br':
            return self.predict_best_responses(xs, return_maximum_gain=True)
        elif self._crit == 'abr':
            return self.approx_max_regret(xs)
        elif self._crit == 'std':
            return -1.0 * self.max_std(xs)

    def suggest_pt(self):
        res_vars = CMAES(self.acq_fct, np.sum(self.n_xs), max_fevals=125 * self.dims,
                         x0=np.random.random(self.dims), is_restart=True).run()
        return res_vars[0], res_vars[1]

    def update_model(self, pt, fs_pt):
        self.pts = pt.reshape(1, -1) if self.pts is None else np.vstack((self.pts, pt))
        for p in range(self.num_players):
            self.fs_pts[p].append(fs_pt[p])
            self.models[p].fit(self.pts, self.fs_pts[p])

    def set_crit(self, crit):
        self._crit = crit

    def step(self):
        """
        one step of the algorithm
        :return:
        """
        xs, _ = self.suggest_pt()
        fs_pt = self.fct(xs)
        self.errs.append(max(np.abs(fs_pt[_p] - self.predict(xs, _p, return_std=False)) for _p in range(self.num_players)))
        self.update_model(xs, fs_pt)

    def run(self):
        for i in range(self._max_fevals):
            if np.random.random() <= self._epsilon:
                self.set_crit('std')
            else:
                self.set_crit('abr')
            self.step()
            print ("iter {} - x: {}, f(x): {}".format(i, self.pts[-1], [self.fs_pts[p][-1] for p in range(self.num_players)]))
            self.disp_log_marginal_likelihood()

    def suggest_approx_ne(self, best_crit='br'):
        """
            This is still experimental and needs further invesitgation. returns the best ne approximation from the sampled points
        :param best_crit: criterion for returning the best approx neq
                `br`: runs best response routine on all the sampled points and returns the one
                      with the minimum regret
                `abr`: returns the point (among the sampled points so far) with the smallest
                        `approx_max_regret`
                `std`: returns the point with the least uncertainty
        """
        regret = np.inf
        best_ne = None
        for pt in self.pts:
            if best_crit == 'br':
                pt_regret = self.predict_best_responses(pt)
            elif best_crit == 'abr':
                pt_regret = self.approx_max_regret(pt)
            elif best_crit == 'std':
                pt_regret = self.max_std(pt)
            else:
                raise Exception("`best_crit` for `suggest_approx_ne` can be either `br` or `abr` or `std`")
            if regret > pt_regret:
                best_ne = pt
                regret = pt_regret

        return best_ne

    def sample_errs(self):
        """
        :return: list of values correspond to the prediction error on the models
        """
        return self.errs

    def disp_log_marginal_likelihood(self):
        for _p in range(self.num_players):
            print("log-marginal likelihood for {}'s payoff: {}".format(_p, self.models[_p].log_marginal_likelihood()))
        print("=" * 10)

    def plot_player_posterior(self, p, title=None, actual_ne=None):
        """
            Plot the posterior mean for p's payoff 
            and the estimate regret. Suitable only for 2D problems
        :param p: player number in range(self.num_players)
        :param title: name of the plot's pdf file (with no extension)
                    if None the plot is shown with `plt.show()`
        :param actual_ne: coordiantes of the actual nash equilibrium
        """
        assert self.dims == 2, "this function can be called for 2-D problems only."
        plot_marginalized_responses(lambda x: self.predict(x, p, return_std=False),
                            lambda x: self.predict_expected_response(np.array([x, 0.1]), p, over_p=True),
                            lambda x: self.predict_expected_response(np.array([0.1, x]), p, over_p=False),
                            lambda x: self.approx_regret(np.array(x), p),
                            p=p,
                            title=title,
                            actual_ne=actual_ne)
