# pylint: disable=too-many-locals,R0801
"""
Example showing how to optimize two Daisy parameters for a single objective using the sequential
optimizer
"""
import argparse
from pathlib import Path
import pandas as pd
from daisypy.optim import (
    DaiFileGenerator,
    DaisySequentialOptimizer,
    ScalarObjective,
    DaisyOptimizationProblem,
    ContinuousParameter,
    DaisyRunner,
    DefaultLogger,
    OutputSpec,
    PostProcessor,
    Simulation,
)

# We want to optimize the sum of squared distance.
# We use the multiprocessing module, which uses pickle, so we cannot use local functions.
def ssd(actual, target):
    '''Sum of squared distance loss function'''
    return ((actual - target)**2).sum()

class SquareOutcome(PostProcessor):
    # pylint: disable=too-few-public-methods
    """Post-process function that squares an outcome

    See daisypy.optim.post_processor.PostProcessor for the interface and
    daisypy.optim.post_processors.Aggregate for a more complicated exampele.
    """
    def __init__(self, outcome_name):
        """
        Parameters
        ----------
        outcome_name : str
          Name of outcome to square
        """
        self.outcome_name = outcome_name

    def __call__(self, outcomes):
        """
        Parameters
        ----------
        outcomes : { str : pandas.DataFrame }
          Dict with named DataFrames. Must have the key `self.outcome_name`

        Returns
        -------
        pandas.DataFrame with columns 'time' and 'value' where 'value' column contains the squared
        outcome.
        """
        df = outcomes[self.outcome_name]
        return pd.DataFrame({"time": df["time"], "value": df["value"]**2})


def run(daisy_path):
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
    out_dir = base_dir / 'out' / 'single-objective-two-parameters-sequential'
    data_dir = base_dir / 'example-data'

    # 0. Define a runner that can run Daisy
    runner = DaisyRunner(daisy_path)

    # 1. Setup the runfile generator
    file_generators = {
        "runfile" : DaiFileGenerator("run.dai", template_file_path=data_dir / "template.dai")
    }

    # Define the simulations outputs.
    outputs = {
        "field" : OutputSpec("field_nitrogen.dlf", 'NO3-Denitrification')
    }

    # Define the simulations
    simulations = {
        "sim" : Simulation(file_generators, outputs)
    }

    # After running simulations we collate the outcomes that we are interested in. An outcome is
    # defined by a name and triplet specifying (simulation, outcome, variable) that uniquely
    # identifies a single column in one of the simulation outputs.
    outcome_specs = {
        "NO3-Denit" : ("sim", "field", "NO3-Denitrification"),
    }

    # If needed we can postprocess the outcomes, for example by squaring the values.
    # The post processing functions are passed a dict with all outcomes
    post_processing = {
        "NO3-Denit_squared" : SquareOutcome("NO3-Denit")
    }

    target = pd.read_csv(data_dir / 'measured-field-nitrogen.csv')
    target["time"] = pd.to_datetime(target[['year', 'month', 'day', 'hour']])

    objective_fn = ScalarObjective(
        name="NO3_Error", # Can be anything
        target=target,
        target_col="NO3-Denitrification", # Must match name in file
        outcome_name="NO3-Denit", # Must match what is in outcomes
        loss_fn=ssd # Function with signature (actual : np.ndarray, target : np.ndarray) -> float
    )

    # 2. Define the parameters that we will optimize
    # Names of parameters should match the names in the template file
    parameters = {
        "runfile" : [
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
        ],
    }


    # 4. Wrap everything as an optimization problem
    # Normally we would not set data_dir and we would set debug = False,
    # but here we set them so we can inspect the output.
    # If debug = False, then outputs are deleted as soon as the optimizer is done with them
    problem = DaisyOptimizationProblem(
        runner,
        simulations,
        outcome_specs,
        post_processing,
        objective_fn,
        parameters,
        data_dir=out_dir / 'data_dir',
        debug=True
    )

    # 5. Setup a logger
    # We use DefaultLogger that logs parameter distributions and sampled parameters to csv files
    log_dir = out_dir / 'logs'
    logger = DefaultLogger(log_dir)

    # 6. Setup an optimizer
    # We use the sequential optimizer.
    options = {
        "num_samples" : 18 # The number of samples to generate from each parameter
    }
    optimizer = DaisySequentialOptimizer(problem, logger, options)

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
    run(args.daisy_path)
