from __future__ import print_function, division
"""
Python wrapper for the GPGame R package.
This can be applied to p >= 2 players
"""
from rpy2.robjects.packages import importr
from rpy2.robjects import r as r_str
from rpy2.robjects import globalenv
from rpy2.robjects import FloatVector
import rpy2.rinterface as ri


class GPGame(object):
    def __init__(self, fct, var_assignment, nobj, max_iters=20, seed=1, n_init=20, grid=31, crit='psim'):
        """
            The current setup o
        :param fct: the function must return a list of floats  denoting the players' cost function (minimization setup)
        :param args: arguments for gpgame

        :param var_assignment: list whose len(var_assignment) denotes the number of total decision variables
                    The value of each entry denotes the player controlling that variable.
                    e.g., var_assignment[0] = 1 (the first payer controls the first variable)
                    len(var_assignment): number of decision variables.
                    max(var_assignment): number of players
        """
        # by defaul the design space is [0,1]^{d_c+d_e}
        self._max_iters = max_iters
        self._var_assignment = var_assignment
        self._nobj = nobj
        assert self._nobj == max(self._var_assignment), "Maybe not all objectives({}) are assigned({})".format(self._nobj, max(self._var_assignment))
        self._seed = seed
        self._n_init = n_init
        assert n_init > len(self._var_assignment), "Anticipate GPGame error. THere must be more intial points({}) than variables({})".format(self._n_init, len(self._var_assignment))
        self._grid = grid
        self._crit = crit
        self.r_fcn_str = '''
                function()
                {{
                    require(GPGame)
                    # Grid definition: player 1 plays x1...xj, player 2 xj+1...xn
                    # The grid is a lattice made of two designs of different sizes
                    n.s <- c({grid_0}, {grid_1})
                    x.to.obj <- c({var_assignment})
                    integcontrol <- list(n.s=n.s, gridtype='{grid_type}')
                    # Set filtercontrol: window filter applied for integration and candidate points
                    filtercontrol <- list(nsimPoints={nsimPoints}, ncandPoints={ncandPoints},
                    filter=c("{filter_type}", "{filter_type}"))
                    # Set km control: lower bound is specified for the covariance range
                    # Covariance type and model trend are specified
                    kmcontrol <- list(lb=rep(.2,4), model.trend=~1, covtype="matern3_2")
                    # Run solver
                    res <- solve_game(fct, equilibrium = "{equilibrium}", crit = "{crit}", n.init={n_init}, n.ite={n_ite},
                    d = {d}, nobj= {nobj}, x.to.obj = x.to.obj,
                    integcontrol={integcontrol},
                    # TODO gives error
                    filtercontrol={filtercontrol},
                    kmcontrol={kmcontrol},
                    ncores = 1, trace={trace}, seed={seed})
                    # Return results
                    res
                 }}
                '''.format(
                    grid_0=self._grid,
                    grid_1=self._grid,
                    grid_type='lhs',
                    filter_type='window',
                    nsimPoints=800,
                    ncandPoints=200,
                    var_assignment=str(self._var_assignment)[1:-1],
                    equilibrium='NE',
                    crit=self._crit,
                    n_init=self._n_init,
                    n_ite = self._max_iters,
                    d=len(self._var_assignment),
                    nobj=self._nobj,
                    integcontrol='integcontrol',
                    filtercontrol='NULL',
                    kmcontrol='NULL',
                    trace=0,
                    seed=self._seed
                )
        print("{} {}".format(self.__class__.__name__, self.__dict__))
        # register the function in the R space
        @ri.rternalize
        def r_fct(x):
            #print(FloatVector(x))
            return FloatVector(fct(x))
        globalenv['fct'] = r_fct
        self._rfunc = r_str(self.r_fcn_str)

    def run(self):
        """
         Runs the GPGame routine and returns the values of the cost functions as well as the
         decision variables per player
        :return: dictionary of the players' decision variables (best) values,
                 the joint decision variables (flattened version of the first),
                 and the corresponding values of
                their cost functions.

        """
        res = self._rfunc()
        for key, val in res.items():
            if key == "Eq.design":
                best_xs_ls = list(val)
                best_xs = {}
                for i in range(self._nobj):
                    best_xs[i] = [best_xs_ls[j] for j, p in enumerate(self._var_assignment) if p == i + 1]
            elif key == "Eq.poff":
                best_j = list(val)

        return best_xs, best_xs_ls, best_j

