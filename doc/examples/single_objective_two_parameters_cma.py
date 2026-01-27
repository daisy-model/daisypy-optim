"""Example showing how to optimize two Daisy parameters for a single objective using CMA"""
import argparse
import os
import pandas as pd
from daisypy.optim import (
    DaiFileGenerator,
    DaisyCMAOptimizer,
    ScalarObjective,
    DaisyOptimizationProblem,
    ContinuousParameter,
    DaisyRunner,
    DefaultLogger
)

# We will use sum of squared distance as the loss
# We use the multiprocessing module, which uses pickle, so we cannot use local functions
def ssd(actual, target):
    return ((actual - target)**2).sum()

def single_objective_two_parameters_cma(daisy_path, daisy_home):
    '''How to optimize parameters for Daisy

    0. Define a runner that can run Daisy
    1. Setup the dai file generator
    2. Define the parameters that we will optimize
    3. Define the objective
    4. Wrap everything as an optimization problem
    5. Setup a logger
    6. Choose an optimizer
    7. Run the optimizer
    8. Look at the results
    '''

    base_out_dir = 'out'

    # 0. Define a runner that can run Daisy
    runner = DaisyRunner(daisy_path, daisy_home)

    # 1. Setup the dai file generator
    dai_template = 'example-data/template.dai'
    dai_file_generator = DaiFileGenerator(template_file_path=dai_template)

    # 2. Define the parameters that we will optimize
    # Names of parameters should match the names in the template file
    parameters = [
        ContinuousParameter(
            name='K_aquitard',
            initial_value=0.2,
            valid_range=(0.1, 0.7)
        ),
        ContinuousParameter(
            name='Z_aquitard',
            initial_value=200,
            valid_range=(150, 250)
        ),
    ]

    # 3. Define the objective
    # We need a target, a loss function and the name of the variable and dlf file
    # The target must be a dataframe with a "time" column
    target = pd.read_csv('example-data/measured-field-nitrogen.csv')
    target["time"] = pd.to_datetime(target[['year', 'month', 'day', 'hour']])

    # The loss function can be any python function mapping a pair of numpy.arrays to a scalar
    loss_fn = ssd

    # We need to know the name of the variable we are optimizing for, and we need to know the name
    # of the Daisy log file where we can find the variable.
    variable_name = "NO3-Denitrification"
    log_name = "field_nitrogen.dlf"
    objective_fn = ScalarObjective(log_name, variable_name, target, loss_fn)


    # 4. Wrap everything as an optimization problem
    # Normally we would not set data_dir and we would set debug = False,
    # but here we set them so we can inspect the output.
    # If debug = False, then outputs are deleted as soon as the optimizer is done with them
    data_dir = os.path.join(base_out_dir, 'data_dir')
    debug = True
    problem = DaisyOptimizationProblem(
        runner, dai_file_generator, objective_fn, parameters, data_dir, debug
    )

    # 5. Setup a logger
    # We use DefaultLogger that logs parameter distributions and sampled parameters to csv files
    log_dir = os.path.join(base_out_dir, 'logs')
    logger = DefaultLogger(log_dir)

    # 6. Setup an optimizer
    # We choose cma. You should try sequential as well.
    # For cma we should always explicitly set the maximum number of function evaluations AKA the
    # maximum number of times we will run Daisy. We set it very low
    cma_options = {
        "maxfevals" : 20
    }
    optimizer = DaisyCMAOptimizer(problem, logger, cma_options)

    # 7. Run the optimizer
    result = optimizer.optimize()

    # 8. Look at the results
    for name, res in result.items():
        print(name)
        for k,v in res.items():
            print('    ', k, ' : ', v, sep='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    parser.add_argument('daisy_home', type=str, help='Path to daisy home directory containing lib/ and sample/')
    args = parser.parse_args()
    single_objective_two_parameters_cma(args.daisy_path, args.daisy_home)
