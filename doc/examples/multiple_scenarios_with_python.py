# pylint: disable=too-many-locals,too-few-public-methods,R0801
"""Example showing how to optimize two Daisy parameters over multiple scenarios"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from daisypy.optim import (
    AggregateObjective,
    DaiFileGenerator,
    PyFileGenerator,
    MultiFileGenerator,
    DaisyCMAOptimizer,
    ScalarObjective,
    DaisyOptimizationProblem,
    ContinuousParameter,
    DaisyRunner,
    DefaultLogger
)

# We use the multiprocessing module, which uses pickle, so we cannot use local functions for loss
# functions and aggregate functions
def mse(actual, target):
    """Mean squared error"""
    return ((actual - target)**2).mean()

class WeightedAverage:
    '''Compute weighted average'''
    def __init__(self, weights):
        self.w = np.array(weights)

    def __call__(self, x):
        return (x * self.w).sum()

def multiple_scenarios(daisy_path):
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
    base_dir = Path(__file__).parent
    out_dir = base_dir / 'out' / 'multiple-scenarios-with-python'
    py_data_dir = base_dir / 'example-data' / 'python-chemical-reaction'
    dai_data_dir = base_dir / 'example-data' / 'multiple-scenarios'

    # 0. Define a runner that can run Daisy
    runner = DaisyRunner(daisy_path)

    # 1. Setup the file generators
    file_generator = MultiFileGenerator({
        'py' : PyFileGenerator('daisy-react.py', template_file_path=py_data_dir / 'daisy-react.py'),
        'dai' : DaiFileGenerator(
            'run.dai', template_file_path=dai_data_dir / 'template-python-reaction.dai'
        ),
    })

    # 2. Define the parameters that we will optimize
    # Names of parameters should match the names in the template files
    parameters = {
        'dai' : [
            ContinuousParameter(
                name='temp_offset',
                initial_value=0,
                valid_range=(-2, 2)
            ),
        ],
        'py' : [ ContinuousParameter('param', 0.24, (0.01, 0.5)) ],
    }

    # 3. Define the objective
    # We have three scenarios. We want to optimize the mean squared error weighted by the number
    # of measurements in each target
    # These where generated with
    # Askov    : temp_offset =  0 (366 samples
    # Jyndevad : temp_offset = -2 (307 samples)
    # Foulum   : temp_offset =  2 (215 samples)
    targets = [
        pd.read_csv(dai_data_dir / 'target-askov.csv'),
        pd.read_csv(dai_data_dir / 'target-jyndevad.csv'),
        pd.read_csv(dai_data_dir / 'target-foulum.csv')
    ]

    # The logs do not have to be the same
    log_names = [
        'askov/field_nitrogen.dlf',
        'jyndevad/field_nitrogen.dlf',
        'foulum/field_nitrogen.dlf',
    ]

    # The variables also dont have to be the same
    variable = 'Matrix-Leaching'
    objective_fns = [
        ScalarObjective(log_name, variable, target, mse) for
        log_name, target in zip(log_names, targets)
    ]

    # Define the weighting
    # We could also do this using sum of squared error as loss_fn and aggregate with mean, but this
    # way illustrates how we can easily define different weights.
    # Just be careful that the order of weights match the order of the targets passed to
    # AggregateObjective
    # This should give temp_offset = -1.5
    weights = np.array(list(map(len, targets)), dtype='float32')
    weights /= weights.sum()

    # You can also try these.
    # only_askov = [1, 0, 0]
    # only_jyndevad = [0, 1, 0]
    # only_foulum = [0, 0, 1]
    # We should get
    #  Askov : temp_offset = 0,
    #  jyndevad : temp_offset = -2
    #  foulum : temp_offset = 2


    # You could also compute weights based on variance of targets
    # This should give temp_offset = -2
    # var_weights = np.array([np.var(target[variable]) for target in targets])
    # var_weights /= var_weights.sum()

    aggregate_fn = WeightedAverage(weights)
    objective = AggregateObjective(objective_fns, aggregate_fn)


    # 4. Wrap everything as an optimization problem
    # Normally we would not set data_dir and we would set debug = False,
    # but here we set them so we can inspect the output.
    # If debug = False, then outputs are deleted as soon as the optimizer is done with them
    out_data_dir = out_dir / 'data_dir'
    debug = True
    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_data_dir, debug
    )

    # 5. Setup a logger
    # We use DefaultLogger that logs parameter distributions and sampled parameters to csv files
    log_dir = out_dir / 'logs'
    logger = DefaultLogger(log_dir)

    # 6. Setup an optimizer
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
    args = parser.parse_args()
    multiple_scenarios(args.daisy_path)
