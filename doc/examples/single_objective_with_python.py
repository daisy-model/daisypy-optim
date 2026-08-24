# pylint: disable=too-many-locals,R0801
"""Example showing how to optimize parameters for python functions used in Daisy"""
import argparse
from pathlib import Path
from daisypy.optim import (
    ContinuousParameter,
    DaiFileGenerator,
    DaisyOptimizationProblem,
    DaisyRunner,
    ScalarObjective,
    DaisyCMAOptimizer,
    DefaultLogger,
    OutputSpec,
    PyFileGenerator,
    Simulation,
    mse
)


def run(daisy_path):
    '''Optimize two parameters in a dai file using a single scalar objective and CMA

    daisy_path: str
      Path to daisy binary
    '''
    base_dir = Path(__file__).parent
    out_dir = base_dir / 'out' / 'single_objective_with_python'
    data_dir = base_dir / 'example-data' / 'python-chemical-reaction'
    file_generators = {
        'py' : PyFileGenerator('daisy-react.py', template_file_path=data_dir / 'daisy-react.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'test-pyreact.dai'),
    }

    outputs = {
        'field' : OutputSpec('soil_NO3_profile.dlf', 'NO3')
    }

    simulations = {
        'sim' : Simulation(file_generators, outputs)
    }

    outcome_specs = {
        'NO3-outcome' : ('sim', 'field', 'NO3'),
    }

    parameters = {
        'py' : [ ContinuousParameter('param', 0.24, (0.01, 0.5)) ],
        'dai' : [ ContinuousParameter('clay', 5, (1, 10)) ],
    }

    runner = DaisyRunner(daisy_path)

    target_file = data_dir / 'target.csv'
    objective = ScalarObjective(
        name='NO3',
        target=target_file,
        target_col='NO3',
        outcome_name='NO3-outcome',
        loss_fn=mse
    )

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, {}, objective, parameters, out_dir
    )
    cma_options = {
        'maxfevals' : 500
    }
    with DefaultLogger(out_dir) as logger:
        optimizer = DaisyCMAOptimizer(problem, logger, cma_options)
        result = optimizer.optimize()

    # Optimum at
    # param = 0.1
    # clay = 2 # (But the effect is very small, so dont expect to get this)
    for name, res in result.items():
        print(name)
        for k, v in res.items():
            print('    ', k, ' : ', v, sep='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    args = parser.parse_args()
    run(args.daisy_path)
