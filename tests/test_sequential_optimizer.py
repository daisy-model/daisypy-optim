import os
import tempfile
from daisypy.optim import (
    CategoricalParameter,
    DefaultLogger,
    DaisySequentialOptimizer,
)
from .mock_problem import MockProblem

def objective(a, b, c):
    return - (a + b + c)

def test_sequential_optimizer(capsys):
    expected_result_log = {
        'step,objective_value,a,b,c',
        '1,-1.0,1.0,0.0,0.0',
        '1,-1.0,0.0,1.0,0.0',
        '1,-2.0,0.0,2.0,0.0',
        '1,-1.0,0.0,0.0,1.0',
        '1,-2.0,0.0,0.0,2.0',
        '1,-3.0,0.0,0.0,3.0',
        '2,-4.0,1.0,0.0,3.0',
        '2,-4.0,0.0,1.0,3.0',
        '2,-5.0,0.0,2.0,3.0',
        '3,-6.0,1.0,2.0,3.0',
    }
    expected_out = '\n'.join([
        'Using at least 11 and at most 15 function evaluations',
        'Evaluating initial parameters',
        'Initial objective = 0',
        'Optimizing',
        'step=1,n_param_sets=6',
        'step=1,total_function_evaluations=7',
        'step=1,best_objective=-3.0',
        'step=1,Fixing c to 3',
        'step=2,n_param_sets=3',
        'step=2,total_function_evaluations=10',
        'step=2,best_objective=-5.0',
        'step=2,Fixing b to 2',
        'step=3,n_param_sets=1',
        'step=3,total_function_evaluations=11',
        'step=3,best_objective=-6.0',
        'step=3,Fixing a to 1',
    ])
    expected_err = ''
    parameters = [
        CategoricalParameter('a', [0,1]),
        CategoricalParameter('b', [0,1,2]),
        CategoricalParameter('c', [0,1,2,3]),
    ]

    problem = MockProblem(parameters, objective)
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisySequentialOptimizer(problem, logger)
            result = optimizer.optimize()
        with open(os.path.join(out_dir, 'result.csv'), 'r', encoding='utf-8') as in_file:
            result_log = { line.strip() for line in in_file }
    assert result_log == expected_result_log

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_out
    assert captured.err.strip() == expected_err

    assert result['a']['best'] == 1
    assert result['b']['best'] == 2
    assert result['c']['best'] == 3
