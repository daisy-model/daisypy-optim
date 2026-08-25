# pylint: disable=too-many-locals,too-few-public-methods,R0801
"""Multi objective optimization over multiple scenarios"""
import argparse
from pathlib import Path
import pandas as pd
from daisypy.optim import (
    ContinuousParameter,
    DaiFileGenerator,
    DaisyAxOptimizer,
    DaisyOptimizationProblem,
    DaisyRunner,
    DefaultLogger,
    MultiObjective,
    OutputSpec,
    ScalarObjective,
    Simulation,
)

# We use the multiprocessing module, which uses pickle, so we cannot use local functions for loss
# functions and aggregate functions

def mse(actual, target):
    """Mean squared error"""
    return ((actual - target)**2).mean()


def run(daisy_path):
    '''Multi objective optimization using DaisyAxOptimizer'''
    base_dir = Path(__file__).parent
    out_dir = base_dir / 'out' / 'multi-objective-ax'
    data_dir = base_dir / 'example-data' / 'multiple-scenarios'

    # 0. Define a runner that can run Daisy
    runner = DaisyRunner(daisy_path)

    # 1. Setup the dai file generator
    file_generators = {
        'dai' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'template.dai')
    }

    # 2. Define the parameters that we will optimize
    # Names of parameters should match the names in the template file
    parameters = [
        ContinuousParameter(
            name='temp_offset',
            initial_value=0,
            valid_range=(-2, 2)
        ),
    ]

    # 3. Define the objective
    # We have three scenarios. These where generated with
    # Askov    : temp_offset =  0 (366 samples)
    # Jyndevad : temp_offset = -2 (307 samples)
    # Foulum   : temp_offset =  2 (215 samples)
    scenarios = ['askov', 'jyndevad', 'foulum']
    targets = {name : pd.read_csv(data_dir / f'target-{name}.csv') for name in scenarios}

    outputs = {
        name : OutputSpec(f'{name}/field_nitrogen.dlf', 'Matrix-Leaching')
        for name in scenarios
    }

    simulations = {
        'sim' : Simulation(file_generators, outputs)
    }

    outcome_specs = {
        name : ('sim', name, 'Matrix-Leaching')
        for name in scenarios
    }

    objective_fns = [
        ScalarObjective(
            name=name,
            target=targets[name],
            target_col='Leaching',
            outcome_name=name,
            loss_fn=mse
        ) for name in scenarios
    ]
    objective = MultiObjective('multi', objective_fns)

    # 4. Wrap everything as an optimization problem
    # Normally we would not set data_dir and we would set debug = False,
    # but here we set them so we can inspect the output.
    # If debug = False, then outputs are deleted as soon as the optimizer is done with them
    problem = DaisyOptimizationProblem(
        runner,
        simulations,
        outcome_specs,
        {},
        objective,
        parameters,
        data_dir=out_dir / 'data_dir',
        debug=True
    )

    # 5. Setup a logger
    # We use DefaultLogger that logs parameter distributions and sampled parameters to csv files
    log_dir = out_dir / 'logs'
    logger = DefaultLogger(log_dir)

    # 6. Setup an optimizer
    options = {
        'max_trials' : 25,
        'max_trials_iteration' : 3
    }
    optimizer = DaisyAxOptimizer(problem, logger, options)
    assert optimizer.multi_objective, 'Optimizer is not multi objective'

    # 7. Run the optimizer
    results = optimizer.optimize()

    # 8. Look at the results
    for result in results:
        for param, value in result.parameters.items():
            print(param, value)
        for metric, value in result.metrics.items():
            print(metric, value)
        print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    args = parser.parse_args()
    run(args.daisy_path)
