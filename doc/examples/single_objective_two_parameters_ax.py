# pylint: disable=too-many-locals,R0801
"""Example showing how to optimize two Daisy parameters for a single objective using CMA"""
import argparse
from pathlib import Path
import pandas as pd
from ax.api.client import Client
from daisypy.optim.ax import daisy_param_to_ax_param
from daisypy.optim import (
    DaiFileGenerator,
    ScalarObjective,
    DaisyOptimizationProblem,
    ContinuousParameter,
    DaisyRunner
)

# We use the multiprocessing module, which uses pickle, so we cannot use local functions
def ssd(actual, target):
    '''Sum of squared distance'''
    return ((actual - target)**2).sum()

def single_objective_two_parameters_ax(daisy_path, daisy_home):
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
    out_dir = base_dir / 'out' / 'single-objective-two-parameters-ax'
    data_dir = base_dir / 'example-data'

    # 0. Define a runner that can run Daisy
    runner = DaisyRunner(daisy_path, daisy_home)

    # 1. Setup the dai file generator
    dai_template = data_dir / 'template.dai'
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
    target = pd.read_csv(data_dir / 'measured-field-nitrogen.csv')
    target["time"] = pd.to_datetime(target[['year', 'month', 'day', 'hour']])

    # The loss function can be any python function mapping a pair of numpy.arrays to a scalar
    loss_fn = ssd

    # We need to know the name of the variable we are optimizing for, and we need to know the name
    # of the Daisy log file where we can find the variable.
    variable_name = "NO3-Denitrification"
    log_name = "field_nitrogen.dlf"
    objective_fn = ScalarObjective("NO3_Error", log_name, variable_name, target, loss_fn)


    # 4. Wrap everything as an optimization problem
    # Normally we would not set data_dir and we would set debug = False,
    # but here we set them so we can inspect the output.
    # If debug = False, then outputs are deleted as soon as the optimizer is done with them
    out_data_dir = out_dir / 'data_dir'
    debug = True
    problem = DaisyOptimizationProblem(
        runner, dai_file_generator, objective_fn, parameters, out_data_dir, debug
    )

    ax_parameters = [ daisy_param_to_ax_param(p) for p in parameters ]
    client = Client()
    client.configure_experiment(parameters=ax_parameters)
    client.configure_optimization(objective=f'-{objective_fn.name}')

    max_trials_total = 50
    max_trials_iteration = 3
    num_trials = 0
    while num_trials < max_trials_total:
        max_trials_this_iteration = min(max_trials_iteration, max_trials_total - num_trials)
        trials = client.get_next_trials(max_trials=max_trials_this_iteration)
        for trial_index, sampled_parameters in trials.items():
            params = [sampled_parameters[p.name] for p in parameters]
            result = problem(params)
            client.complete_trial(trial_index=trial_index, raw_data=result)
        num_trials += len(trials)

    best_parameters, prediction, index, name = client.get_best_parameterization()
    print("Best Parameters:", best_parameters)
    print("Prediction (mean, variance):", prediction)
    print('index, name', index, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    parser.add_argument('--daisy_home', type=str, default=None,
                        help='Path to daisy home directory containing lib/ and sample/')
    args = parser.parse_args()
    single_objective_two_parameters_ax(args.daisy_path, args.daisy_home)
