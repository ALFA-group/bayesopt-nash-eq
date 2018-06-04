import os
import json
import sys
import argparse
import multiprocessing
import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor as PPE, as_completed

from ne.nash_eq_algs.bayes_ne import BayesNE
from ne.nash_eq_algs.stoch_search_methods import iterated_best_response, hier_sa
from ne.nash_eq_algs.gpgame import GPGame

from ne.benchmark.saddle import Saddle
from ne.benchmark.mop1 import MOP1
from ne.benchmark.adnet_game import AdnetGame
import c_implementation.adnet as adnet


def run_saddle_unit_exp(setup):
    """
    :param setup:
    :return:
    """
    dim = setup['dim']
    run = setup['run']
    alg = setup['alg']
    alg_variant = setup['variant']
    fevals = setup['fevals']
    kwargs = setup['kwargs']
    if alg == 'BN':
        problem = Saddle(is_minimize=False, dim=dim)
        bneq = BayesNE(problem.evaluate, [dim // 2, dim // 2], max_fevals=fevals, seed=run, **kwargs)
        bneq.run()
        #best_xs = bneq.best_approx_neq()
    elif alg == 'BR':
        problem = Saddle(dim=dim)
        best_xs = iterated_best_response(problem.evaluate, [dim // 2, dim // 2], max_fevals=fevals, seed=run)
    elif alg == 'SS':
        problem = Saddle(dim=dim)
        best_xs = hier_sa(problem.evaluate, [dim // 2, dim // 2], max_fevals=fevals, seed=run)
    elif alg == 'GPG':
        problem = Saddle(dim=dim)
        var_assignment = [1] * (dim // 2) + [2] * (dim // 2)
        n_init = max(dim + 1, 0.25 * fevals)
        max_iters = fevals - n_init
        grid = 31 * (dim // 2)
        gpgame = GPGame(problem.evaluate, var_assignment=var_assignment, nobj=2, max_iters=max_iters,
                        grid=grid, n_init=n_init, seed=run, **kwargs)
        print('Run GPGame on simple function')
        _, best_xs, best_j = gpgame.run()
    else:
        raise Exception("No such algorithm")

    # compute regret trace
    actual_fevals = problem.get_num_evals()
    best_regret = np.inf
    regret_trace = []
    for xs in problem.get_xs()[:fevals]:
        best_regret = min(best_regret, problem.ne_regret(xs))
        regret_trace.append(best_regret)


    stats = {
        'fevals': fevals,
        'run': run,
        'alg': alg_variant,
        'dim': dim,
        'actual_fevals': actual_fevals,
        'regret_trace': regret_trace
    }
    print(stats)

    with open(os.path.join(os.path.dirname(__file__),
                           setup['result_path'][:-5] +
                                   '_{alg}_{dim}_{run}.json'.format(alg=alg_variant, dim=dim, run=run))
            , 'w') as f:
        json.dump(stats, f, sort_keys=True, indent=4)
    return stats



def run_mop_unit_exp(setup):
    """
    :param setup:
    :return:
    """
    dim = setup['dim']
    run = setup['run']
    alg = setup['alg']
    alg_variant = setup['variant']
    fevals = setup['fevals']
    kwargs = setup['kwargs']
    assert dim == 2, "mop dim should be 2"
    if alg == 'BN':
        problem = MOP1(is_minimize=False)
        bneq = BayesNE(problem.evaluate, [dim // 2, dim // 2], max_fevals=fevals, seed=run, **kwargs)
        bneq.run()
        #best_xs = bneq.best_approx_neq()
    elif alg == 'BR':
        problem = MOP1()
        best_xs = iterated_best_response(problem.evaluate, [dim // 2, dim // 2], max_fevals=fevals, seed=run)
    elif alg == 'SS':
        problem = MOP1()
        best_xs = hier_sa(problem.evaluate, [dim // 2, dim // 2], max_fevals=fevals, seed=run)
    elif alg == 'GPG':
        problem = MOP1()
        var_assignment = [1] * (dim // 2) + [2] * (dim // 2)
        n_init = max(dim + 1, 0.25 * fevals)
        max_iters = fevals - n_init
        grid = 31 * (dim // 2)
        gpgame = GPGame(problem.evaluate, var_assignment=var_assignment, nobj=2, max_iters=max_iters,
                        grid=grid, n_init=n_init, seed=run, **kwargs)
        print('Run GPGame on simple function')
        _, best_xs, best_j = gpgame.run()
    else:
        raise Exception("No such algorithm")

    # compute regret trace
    actual_fevals = problem.get_num_evals()
    best_regret = np.inf
    regret_trace = []
    for xs in problem.get_xs()[:fevals]:
        best_regret = min(best_regret, problem.ne_regret(xs))
        regret_trace.append(best_regret)


    stats = {
        'fevals': fevals,
        'run': run,
        'alg': alg_variant,
        'dim': dim,
        'actual_fevals': actual_fevals,
        'regret_trace': regret_trace
    }
    print(stats)

    with open(os.path.join(os.path.dirname(__file__),
                           setup['result_path'][:-5] +
                                   '_{alg}_{dim}_{run}.json'.format(alg=alg_variant, dim=dim, run=run))
            , 'w') as f:
        json.dump(stats, f, sort_keys=True, indent=4)
    return stats


def run_experiment(config_file):
    with open(config_file) as f:
        config = yaml.load(f)


    pool = PPE()
    fs = []
    results = []

    for dim in config['dimensions']:
        for run in range(config['num_runs']):
            for fevals in config['fevals_per_dim']:
                for alg in config['algs']:
                    for variant in config[alg]:
                        setup = {
                            'dim': dim,
                            'fevals': min(80, fevals * dim),
                            'run': run + config['runs_offset'],
                            'variant': variant['name'],
                            'alg': alg,
                            'kwargs': variant['kwargs'],
                            'result_path': config['result_path']
                        }
                        if config['name'] == 'saddle':
                            fs.append(pool.submit(run_saddle_unit_exp, setup))
                        elif config['name'] == 'mop':
                            fs.append(pool.submit(run_mop_unit_exp, setup))
                        elif config['name'] == 'adhd':
                            if run == 0:
                                try:
                                    bkp_file = os.path.join(os.path.dirname(__file__), config['result_path'].replace('.json', '_bkp.json'))
                                    os.remove(bkp_file)
                                except FileNotFoundError:
                                    pass

                            setup['n_init'] = config['n_init']
                            setup['start_seed'] = config['start_seed']
                            setup['result_path_bkp'] = bkp_file
                            fs.append(pool.submit(run_adhd_unit_exp, setup))
                        else:
                            raise Exception('unknown experiment')

    results = []
    for x in as_completed(fs):
        try:
            results.append(x.result())
        except Exception as e:
            print(e)

    print(results)
    with open(os.path.join(os.path.dirname(__file__), config['result_path']), 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


def parse_args(args):
    """
    Parse command line arguments (`sys.argv`).
    :return: settings from configuration file and CLI arguments
    :rtype dict:
    """
    parser = argparse.ArgumentParser(description="Run NE experiments",
                                     argument_default=None)
    parser.add_argument(
        "-f",
        "--configuration_file",
        type=str,
        required=True,
        dest="configuration_file",
        help="YAML configuration file. E.g. "
        "configurations/demo_gpgame.yml")

    return parser.parse_args(args=args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file = os.path.join(os.path.dirname(__file__), "configs/saddle_config.yml")
    else:
        args = parse_args(sys.argv[1:])
        config_file = args.configuration_file
        
    run_experiment(config_file)

