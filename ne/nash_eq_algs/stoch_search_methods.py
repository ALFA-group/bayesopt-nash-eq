"""
Python implementation of methods described in the paper:

Stochastic Search Methods for Nash Equilibrium
Approximation in Simulation-Based Games,
Vorobeychik, Wellman

@article{vorobeychik2010probabilistic,
  title={Probabilistic analysis of simulation-based games},
  author={Vorobeychik, Yevgeniy},
  journal={ACM Transactions on Modeling and Computer Simulation (TOMACS)},
  volume={20},
  number={3},
  pages={16},
  year={2010},
  publisher={ACM}
}


NOTE: the implementations here are for zero-sum two-player with stochastic functions
"""
from __future__ import division
import numpy as np
import random
from scipy.optimize import basinhopping
from ne.minimizers.cmaes import CMAES
from ne.utils.plots import plot_marginalized_responses, plot_decision_space, plot_objective_space


def best_response(fct, xs0, n_xs, max_iters=10, max_fevals=10):
    """
    Compute the best response (deviation) for both player x and player y given y0 and x0, respectively.
    # TODO the current setup only supports two players whose decision variables
    # are ordered i.e. xs0 = [x0,...xn_xs[0], x0, ..., xn_xs[1]] player 1 vars then player 2 vars
    :param fct: vectorial payoff function f:X^{sum(n_xs)} --> R^{num_players}
    :param xs0: joint decision variable of the players
                It is assumed that all the decision var are in [0,1]
    :param n_xs: number of variables for each player
                    e.g., player 1 has n_xs[0] decision variables
                    It is assumed that all the decision var are in [0,1]
    :param max_fevals: This is currently has no effect
    :return: best x given y0, best y given x0
    """
    f_vals = fct(xs0)

    def f_x(x):
        xs = xs0.copy()
        xs[:n_xs[0]] = x
        return fct(xs)[0]

    def f_y(y):
        xs = xs0.copy()
        xs[n_xs[0]:] = y
        return fct(xs)[1]


    # best response
    res_x = basinhopping(f_x, xs0[:n_xs[0]], niter=max_iters,
                         minimizer_kwargs={'method': 'L-BFGS-B',
                                           'bounds': [(0, 1)] * n_xs[0],
                                           'options': {'maxfun': max_fevals // 2}})
    res_y = basinhopping(f_y, xs0[n_xs[0]:], niter=max_iters,
                         minimizer_kwargs={'method': 'L-BFGS-B',
                                           'bounds': [(0, 1)] * n_xs[1],
                                           'options': {'maxfun': max_fevals // 2}})

    _x = res_x.x
    _y = res_y.x

    gain_x = f_vals[0] - res_x.fun
    gain_y = f_vals[1] - res_y.fun

    return _x, _y, gain_x, gain_y

def iterated_best_response(fct, n_xs, max_fevals, is_verbose=True, seed=1):
    """
        Algorithm 1 of the paper
    TODO: extends fct to multi-player setup
    :param fct: vectorial payoff function f:X^{sum(n_xs)} --> R^{num_players}
    :param xs0: joint decision variable of the players
                It is assumed that all the decision var are in [0,1]
    :param n_xs: number of variables for each player
                    e.g., player 1 has n_xs[0] decision variables
                    It is assumed that all the decision var are in [0,1]
    :param max_fevals: maximum number of function evaluations
    :return: solutions for neq
    """
    # set seed
    np.random.seed(seed)
    random.seed(seed)

    # initialize solution
    _xs0 = np.random.random(sum(n_xs))

    # set iters of the method, iters of best_response method, evals per best_response iter
    num_iters = 5
    num_br_iters = 2
    num_evals_per_br_iter = max(1, max_fevals // (num_iters * num_br_iters))

    # main routine
    for _ in range(num_iters):
        _x, _y, gain_x, gain_y = best_response(fct, _xs0, n_xs, max_iters=num_br_iters, max_fevals=num_evals_per_br_iter)
        print("x's gain:{}, y's gain:{}".format(gain_x, gain_y))

    return np.concatenate([_x, _y])


def hier_sa(fct, n_xs, max_fevals=100, is_verbose=True, seed=1):
    """
     Algorithm 2 of the paper
    :param fct: vectorial payoff function f:X^{sum(n_xs)} --> R^{num_players}
    :param n_xs: number of variables for each player
                    e.g., player 1 has n_xs[0] decision variables
                    It is assumed that all the decision var are in [0,1]
    :param max_fevals: function evaluation budget
    :param is_verbose:
    :return:
    """
    # set seed
    np.random.seed(seed)
    random.seed(seed)
    dims = sum(n_xs)
    outer_iters = 5
    inner_iters = 5
    inner_fevals = int(0.2 * max_fevals)
    outer_fevals = max(1, max_fevals // (outer_iters * inner_iters * inner_fevals))

    def hier_fct(s):
        """
        :param s: joint x, y
        :return:
        """
        _, _, gain_x, gain_y = best_response(fct, s, n_xs, max_iters=inner_iters, max_fevals=inner_fevals)
        return max(gain_x, gain_y)

    res = basinhopping(hier_fct, np.random.random(dims), niter=outer_iters,
                       minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': [(0, 1)] * dims, 'options': {'maxfun': outer_fevals}})

    return res.x













