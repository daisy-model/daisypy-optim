# daisypy-optim
Framework for optimization of Daisy parameters.

## Overview
The idea of the framework is to setup an isolated, self-contained environment where the optimization can be run. The optimization is defined by a
* template dai file
* python script

A tool is included that generates the python script based on user input. Once generated, it can be adapted to specific needs.

There are several optimization methods available

 * `sequential` : A greedy approach that fixes one parameter at a time. Works well when optimizing few paramaters.
 * `cma` : CMA-ES. Works well for larger number of parameters and when the outcome depends strongly on parameter combinations.
 * `skopt` : scikit-optimize.

Both continuous and categorical parameters are supported. TODO: Document how they are treated for each optimizer.

## Getting started

* [Windows](doc/getting-started-win.md)
* [Linux](doc/getting-started-win.md)
* [Examples](doc/examples)

If you do not want to read a getting started guide, you can just install with

    pip install daisypy-optim@git+https://github.com/daisy-model/daisypy-optim

To enable CMA-ES

    pip install cma

To enable scikit-optimize

    pip install scikit-optimize joblib

To enable Tensorboard logging

    pip install tensorboard torch scipy

All of the above

    pip install daisypy-optim@git+https://github.com/daisy-model/daisypy-optim cma scikit-optimize joblib

