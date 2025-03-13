"""Example showing how to do parameter optimization in Daisy"""
import pandas as pd
from daisypy.optim import (
    DaisyLoss,
    DaisyObjective,
    DaisyRunner,
    DaiFileGenerator
)

def main():
    # Define the parameters that we want to optimize
    # 1. Define the template dai file
    # 2. Setup the dai file generator
    # 3. Define the parameters that we will optimize

    # Define the template dai file
    dai_template = 'example-data/template.dai'

    # Setup the dai file generator
    dai_file_generator = DaiFileGenerator(dai_template)

    # Define the parameters we want to optimize
    
    dai_file = dai_file_generator('example-data/out', { 'K_aquitard' : 0.3 })
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

if __name__ == '__main__':
    main()
#     runner = DaisyRunner(
#         '/home/silas/Projects/daisy-model/daisy/build/portable/daisy',
#         '/home/silas/Projects/daisy-model/daisy'
#     )
#     runner('/home/silas/Projects/daisy-model/daisypy-optim/sample.dai',
#            '/home/silas/Projects/daisy-model/daisypy-optim/tmp',

# if __name__ == '__main__':
#     objective = DaisyObjective("field_nitrogen.dlf", "NO3-Denitrification", 0, sum_squared_distance)
#     objective("/home/silas/Projects/daisy-model/daisypy-optim/tmp")
           

            
# myDai = DaiFileGenerator('template.txt')
 
# myDai('test',lai_folder= 'blabla', file_name='my_file', output_folder='outie', Pixel_nr='2')           
