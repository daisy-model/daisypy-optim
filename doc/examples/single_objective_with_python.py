# pylint: disable=too-many-locals
"""Example showing how to optimize parameters for python functions used in Daisy"""
import argparse
from pathlib import Path
from daisypy.optim import (
    ContinuousParameter,
    DefaultLogger,
    DaisyCMAOptimizer,
    PyFileGenerator,
    DaiFileGenerator,
    MultiFileGenerator,
    DaisyRunner,
    DaisyOptimizationProblem,
    ScalarObjective,
    mse
)


def single_objective_with_python(daisy_path, daisy_home=None):
    '''Optimize two parameters in a dai file using a single scalar objective and CMA

    daisy_path: str
      Path to daisy binary

    daisy_home: str
      Path to daisy home. If None let the Daisy binary figure it out
    '''    
    base_dir = Path(__file__).parent
    out_dir = base_dir / 'out' / 'single_objective_with_python'
    data_dir = base_dir / 'example-data' / 'python-chemical-reaction'
    file_generator = MultiFileGenerator({
        'py' : PyFileGenerator('daisy-react.py', template_file_path=data_dir / 'daisy-react.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'test-pyreact.dai'),
    })

    parameters = {
        'py' : [ ContinuousParameter('param', 0.24, (0.01, 0.5)) ],
        'dai' : [ ContinuousParameter('clay', 5, (1, 10)) ],
    }

    runner = DaisyRunner(daisy_path, daisy_home)

    target_file = data_dir / 'target.csv'
    objective = ScalarObjective("soil_NO3_profile.dlf", "NO3", target_file, mse)

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_dir
    )
    cma_options = {
        "maxfevals" : 500
    }
    with DefaultLogger(out_dir) as logger:
        optimizer = DaisyCMAOptimizer(problem, logger, cma_options)
        result = optimizer.optimize()

    for name, res in result.items():
        print(name)
        for k,v in res.items():
            print('    ', k, ' : ', v, sep='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    parser.add_argument('--daisy-home', type=str,
                        help='Path to daisy home directory containing lib/ and sample/',
                        default=None)
    args = parser.parse_args()
    single_objective_with_python(args.daisy_path, args.daisy_home)
