# bayesopt-nash-eq

Code repository for [Approximating Nash Equilibria for Black-Box
Games: A Bayesian Optimization Approach](https://arxiv.org/pdf/1804.10586.pdf)


## Python environment

```
conda install nb_conda
conda config --add channels conda-forge
conda env create --file environment.yml
```

This will create an environment called ne.

To activate this environment, execute:

```
source activate ne
```

## Running the demo

The `demo.ipynb` demonstrates the algorithm along with the algorithms considered in the paper. It also demonstrates other utils/plots that can be found in the repo.

```
jupyter notebook
```

## Running Experiments

The script `toy_experiments.py` performs experiments on `SADDLE` and `MOP` problems. The experiment can be configured according to a configuration files, as follows. `cd` to the repo directory and do the following:

```
export PYTHONPATH=.
python ne/experiments/toy_experiments.py --file ne/experiments/configs/saddle_config.yml
```

As the experiment is running, results of different runs/algorithms/problems will be stored in `ne/experiments/res` as `{experiment_name}_{alg_name}_{dimension}_{run_number}.json`, these files are helpful for backup/monitoring purposes. At the end of the experiment a json file `{experiment_name}.json` will be generated which essentially concatentates all `{experiment_name}_*.json`

## Plotting Results

The `demo.ipynb` demonstrates how the results can be plotted. Moreover, a `json` file whose format is similar to that created by `toy_experiments.py` can be passed to the `plot_regret_trace` function under `ne/utils/plots.py` as demonstrated in the `main` block of `ne/utils/plots.py`.