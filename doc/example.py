"""Example showing how to do parameter optimization in Daisy"""
import pandas as pd
from daisypy.optim import (
    DaiFileGenerator,
    DaisyLoss,
    DaisyObjective,
    DaisyOptimizationProblem,
    DaisyParameter,
    DaisyRunner,
)

def main():
    # Setup Daisy so we can run it
    daisy_path = '/home/silas/Projects/daisy-model/daisy/build/portable/daisy' #<path-to-daisy-binary>
    daisy_home = '/home/silas/Projects/daisy-model/daisy' #<path-to-daisy-home-directory>
    runner = DaisyRunner(daisy_path, daisy_home)

    # Test it
    runner('', 'example-data/out')
    with open('example-data/out/daisy.log', 'r', encoding='utf-8') as file:
        print(''.join(file))

    # Define the parameters that we want to optimize
    # 1. Define the template dai file
    # 2. Setup the dai file generator
    # 3. Define the parameters that we will optimize

    # Define the template dai file
    dai_template = 'example-data/template.dai'

    # Setup the dai file generator
    dai_file_generator = DaiFileGenerator(dai_template)

    # Define the parameters we want to optimize
    parameters = [
        DaisyParameter(
            name='K_aquitard',
            initial_value=0.2,
            valid_range=(0.1, 0.7)
        )
    ]

    # Test that it works
    sampled_parameters = { p.name : p.initial_value for p in parameters }
    dai_file = dai_file_generator('example-data/out', sampled_parameters)
    with open(dai_file, encoding='utf-8') as file:
        print(''.join(file))

    # Define the objective that we want to optimize
    # 1. Define the target
    # 2. Define the loss function
    # 3. Define the objective

    # The target must be a dataframe with a "time" column
    target = pd.read_csv('example-data/measured-field-nitrogen.csv')
    target["time"] = pd.to_datetime(target[['year', 'month', 'day', 'hour']])

    # We will use sum of squared distance as the loss
    def ssd(actual, target):
        return ((actual - target)**2).sum()

    # We need to wrap the loss function with the DaisyLoss class
    loss_fn = DaisyLoss(ssd)

    # We need to know the name of the variable we are optimizing for, and we need to know the name
    # of the Daisy log file where we can find the variable.
    variable_name = "NO3-Denitrification"
    log_name = "field_nitrogen.dlf"
    objective_fn = DaisyObjective(log_name, variable_name, target, loss_fn)

    # We can now compute the value of the objective by calling the objective function with a path to
    # a directory containing a log file
    print(objective_fn("example-data"))

    problem = DaisyOptimizationProblem(runner, dai_file_generator, objective_fn, parameters)

    # Test it
    # Use the center of the valid ranges
    parameter_values = [sum(p.valid_range) / 2 for p in parameters]
    result = problem(parameter_values)
    print(result)

if __name__ == '__main__':
    main()
