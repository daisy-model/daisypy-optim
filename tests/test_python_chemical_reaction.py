from pathlib import Path
from daisypy.optim import (
    CategoricalParameter,
    DefaultLogger,
    DaisySequentialOptimizer,
    PyFileGenerator,
    DaiFileGenerator,
    MultiFileGenerator,
    DaisyRunner,
    DaisyOptimizationProblem,
    ScalarObjective,
    mse
)


def test_python_chemical_reaction(tmp_path):
    '''Test that we can optimize parameters of a python function in a full sim setup'''
    data_dir = Path(__file__).parent / 'test-data' / 'python-chemical-reaction'
    file_generator = MultiFileGenerator({
        'py' : PyFileGenerator('daisy-react.py', template_file_path=data_dir / 'daisy-react.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'test-pyreact.dai'),
    })

    parameters = {
        'py' : [ CategoricalParameter('param', [0.24, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) ],
        'dai' : [],
    }

    # Assume we are on linux and daisy is installed
    runner = DaisyRunner('daisy')

    target_file = data_dir / 'target.csv'
    objective = ScalarObjective("soil_NO3_profile.dlf", "NO3", target_file, mse)

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, tmp_path, debug=True
    )
    with DefaultLogger(tmp_path) as logger:
        optimizer = DaisySequentialOptimizer(problem, logger)
        result = optimizer.optimize()

    assert result['param']['best'] == 0.1
