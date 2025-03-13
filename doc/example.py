"""Example showing how to do parameter optimization in Daisy"""
import argparse
import pandas as pd
from daisypy.optim import (
    DaiFileGenerator,
    DaisyCMAOptimizer,
    DaisyLoss,
    DaisyObjective,
    DaisyOptimizationProblem,
    DaisyParameter,
    DaisyRunner,
)

# We will use sum of squared distance as the loss
# We use the multiprocessing module, which uses pickle, so we cannot use local functions
def ssd(actual, target):
    return ((actual - target)**2).sum()

def example(daisy_path, daisy_home):
    # How to optimize parameters for Daisy    
    # 0. Define a runner that can run Daisy
    # 1. Define the template dai file
    # 2. Setup the dai file generator
    # 3. Define the parameters that we will optimize
    # 4. Define the target
    # 5. Wrap a loss function
    # 7. Define the objective
    # 8. Wrap everything as an optimization problem
    # 9. Choose an optimizer
    # 10. Run the optimizer
    # 11. Get the results

    # Setup Daisy so we can run it
    runner = DaisyRunner(daisy_path, daisy_home)
    
    # Names of parameters should match the names in the template file
    dai_template = 'example-data/template.dai'
    dai_file_generator = DaiFileGenerator(dai_template)
    parameters = [
        DaisyParameter(
            name='K_aquitard',
            initial_value=0.2,
            valid_range=(0.1, 0.7)
        ),
        DaisyParameter(
            name='Z_aquitard',
            initial_value=200,
            valid_range=(150, 250)
        ),
    ]

    # The target must be a dataframe with a "time" column
    target = pd.read_csv('example-data/measured-field-nitrogen.csv')
    target["time"] = pd.to_datetime(target[['year', 'month', 'day', 'hour']])

    # We need to wrap the loss function with the DaisyLoss class
    loss_fn = DaisyLoss(ssd)

    # We need to know the name of the variable we are optimizing for, and we need to know the name
    # of the Daisy log file where we can find the variable.
    variable_name = "NO3-Denitrification"
    log_name = "field_nitrogen.dlf"
    objective_fn = DaisyObjective(log_name, variable_name, target, loss_fn)

    # Setup the optimization problem
    problem = DaisyOptimizationProblem(runner, dai_file_generator, objective_fn, parameters)

    # Choose an optimizer
    # We choose cma (because we dont have anything else atm...)
    # For cma we should always explicitly set the maximum number of function evaluations AKA the
    # maximum number of times we will run Daisy.
    cma_options = {
        "maxfevals" : 20
    }
    optimizer = DaisyCMAOptimizer(problem, cma_options)

    # Optimize and print the result
    result = optimizer.optimize()
    for name, res in result.items():
        print(name)
        for k,v in res.items():
            print('    ', k, ' : ', v, sep='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    parser.add_argument('daisy_home', type=str, help='Path to daisy home directory containing lib/ and sample/')
    args = parser.parse_args()
    example(args.daisy_path, args.daisy_home)
