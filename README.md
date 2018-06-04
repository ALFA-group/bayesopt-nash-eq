# bayesopt-nash-eq

Code repository for [Approximating Nash Equilibria for Black-Box
Games: A Bayesian Optimization Approach](https://arxiv.org/pdf/1804.10586.pdf)


## Python environment

```
conda install nb_conda
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

The `demo.ipynb` demonstrates how the results can be plotted. Moreover, a `json` file whose format is similar to that created by `toy_experiments.py` can be passed to the `plot_regret_trace` function under `ne/utils/plots.py` as demonstrated in the `main` block of `ne/utils/plots.py`. The results of the `saddle_config.yml` experiment is stored
in `ne/experiments/res/saddle_res.json` 

For the experiment defined above, the `plots.py` script is set to display its result:

```
python ne/utils/plots.py
```


## Running GPGame from Python

There might be some difficulty in installing the "GPGame" package and interfacing it with Python. Make sure you install the package (and all the required packages) with `sudo`. Then copy it to the environment's `R/library` 

```
sudo R
>> install.packages("GPGame")
>> quit()
sudo cp -a ~/R/x86_64-pc-linux-gnu-library/3.2/. ~/anaconda2/envs/ne/lib/R/library/
```

## Running Multithreaded experiments

There is unintended multithreading with numpy ( https://stackoverflow.com/questions/19257070/unintented-multithreading-in-python-scikit-learn )

Check the blas/lapack library used and set the number of threads. E.g.

```
export OPENBLAS_NUM_THREADS=1
```