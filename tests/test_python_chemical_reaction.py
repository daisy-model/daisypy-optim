from pathlib import Path
from daisypy.optim import (
    CategoricalParameter,
    DaiFileGenerator,
    DaisyOptimizationProblem,
    DaisyRunner,
    DaisySequentialOptimizer,
    DefaultLogger,
    OutputSpec,
    PyFileGenerator,
    ScalarObjective,
    Simulation,
    mse,
)
from .markers import requires_daisy

@requires_daisy
def test_python_chemical_reaction(tmp_path):
    '''Test that we can optimize parameters of a python function in a full sim setup'''
    data_dir = Path(__file__).parent / 'test-data' / 'python-chemical-reaction'
    file_generators = {
        'py' : PyFileGenerator('daisy-react.py', template_file_path=data_dir / 'daisy-react.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'test-pyreact.dai'),
    }

    outputs = {
        "field" : OutputSpec("soil_NO3_profile.dlf", "NO3")
    }

    simulations = {
        "sim" : Simulation(file_generators, outputs)
    }

    outcome_specs = {
        "NO3-outcome" : ("sim", "field", "NO3"),
    }


    parameters = {
        'py' : [ CategoricalParameter('param', [0.24, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) ],
        'dai' : [],
    }

    # Assume we are on linux and daisy is installed
    runner = DaisyRunner('daisy')

    target_file = data_dir / 'target.csv'
    objective = ScalarObjective(
        name="NO3",
        target=target_file,
        target_col="NO3",
        outcome_name="NO3-outcome",
        loss_fn=mse
    )

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, {}, objective, parameters, tmp_path, debug=True
    )
    with DefaultLogger(tmp_path) as logger:
        optimizer = DaisySequentialOptimizer(problem, logger)
        result = optimizer.optimize()

    assert result['param']['best'] == 0.1
