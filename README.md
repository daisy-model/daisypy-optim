# Optimization framework for [Daisy](https://github.com/daisy-model/daisy)

A python based framework for optimizing parameters in [Daisy]((https://github.com/daisy-model/daisy). 

## Features
* Multiple optimization methods: a greedy sequential optimizer, CMA-ES, and Bayesian optimizers
* Optimization across multiple scenarios
* Single or multi-objective optimization
* Optimization of parameters in both Daisy (`.dai`) and Python (`.py`) files
* Support for categorical and continuous parameters (depending on optimizer)


## Getting started
The framework is available on https://pypi.org/ and can be installed with pip

```
pip install daisypy-optim
```

By default this will only install the greedy sequential optimizer. For most problems you will want to use either CMA-ES or a Bayesian optimizer. The CMA-ES optimizer can be installed with
```
pip install daisypy-optim[cma]
```

and the Bayesian optimizers with
```
pip install daisypy-optim[ax]
```
Note that the Bayesian optimizers pulls in a lot of dependencies.

## Examples
There are several examples illustrating how to optimize parameters in various situations. They all share a common structure

1. Setup a `DaisyRunner` that knows how to run Daisy
2. Setup a `FileGenerator` that can generate `.dai` and `.py`
3. Define the parameters that should be optimized for
4. Define the objective that should be optimized
5. Setup and run an optimizer

See [doc/examples](doc/examples] for an overview of the examples.
