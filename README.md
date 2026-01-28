# daisypy-optim
Framework for optimization of Daisy parameters.

## Overview
The framework can be used in two ways

1. As a self-contained environment
2. As a regular python module

### Self contained optimization environment
The idea of self-contained environment is to create a repository that contains everything that is needed for the optimization (except the Daisy binary). This allow easy sharing, version control and archiving. This type of optimization is defined by

* a single dai file template
* zero or more python file templates
* a python script that runs the simulation

A tool is included that generates the python script based on user input. Once generated, it can be adapted to specific needs. To start using `daisypy-optim` this way look at

* [Windows](doc/getting-started-win.md)
* [Linux](doc/getting-started-linux.md)


### Regular python modules
It is also possible to use `daisypy-optim` as a regular python module. Simply install with

    pip install daisypy-optim@git+https://github.com/daisy-model/daisypy-optim

To enable CMA-ES

    pip install cma

To enable scikit-optimize

    pip install scikit-optimize joblib

All of the above

    pip install daisypy-optim@git+https://github.com/daisy-model/daisypy-optim cma scikit-optimize joblib


And look at the [examples](doc/examples) and [tests](tests) to see how to setup an optimization

## Chossing an optimization method
There are several optimization methods available

 * `sequential` : A greedy approach that fixes one parameter at a time. Works well when optimizing few paramaters.
 * `cma` : CMA-ES. Works well for larger number of parameters and when the outcome depends strongly on parameter combinations. Does not work for single parameters.
 * `skopt` : scikit-optimize. Bayesian optimization. Very slow. Will be replaced with a BoTorch/Ax method.


## Continuous/categorical parameters
Both continuous and categorical parameters are supported. Currently, the sequential method treats everything as categorial parameters (by sampling a fixed number of values from the continuous parameters). The other methods requires continuous parameters.
