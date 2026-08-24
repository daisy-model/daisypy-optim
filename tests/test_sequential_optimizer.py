# pylint: disable=relative-beyond-top-level
import os
import tempfile
import pytest
import pandas as pd
from daisypy.optim import (
    CategoricalParameter,
    DefaultLogger,
    DaisySequentialOptimizer,
)
from .mockup import MockProblem, MockError, ProblemFailAfterN

class Objective():
    # pylint: disable=too-few-public-methods
    '''Negative value of outcome'''
    def __init__(self, name):
        self.name = name
        self.outcome_name = "outcome"
        self.target = pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01']),
            'value' : [42]
        })

    def __call__(self, a, b, c):
        return - (a + b + c)

def test_sequential_optimizer(capsys):
    '''Test that sequential optimizer finds the optimmum and generates expected output'''
    # pylint: disable=too-many-locals
    expected_samples_log = [
        'evaluation_id,step,tag,metric_neg_sum,param_a,param_b,param_c',
        '"1:0",1,"raw",-1.0,1.0,0.0,0.0',
        '"1:1",1,"raw",-1.0,0.0,1.0,0.0',
        '"1:2",1,"raw",-2.0,0.0,2.0,0.0',
        '"1:3",1,"raw",-1.0,0.0,0.0,1.0',
        '"1:4",1,"raw",-2.0,0.0,0.0,2.0',
        '"1:5",1,"raw",-3.0,0.0,0.0,3.0',
        '"2:0",2,"raw",-4.0,1.0,0.0,3.0',
        '"2:1",2,"raw",-4.0,0.0,1.0,3.0',
        '"2:2",2,"raw",-5.0,0.0,2.0,3.0',
        '"3:0",3,"raw",-6.0,1.0,2.0,3.0',
    ]
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

    problem = MockProblem(parameters, Objective("neg_sum"))
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisySequentialOptimizer(problem, logger)
            result = optimizer.optimize()
        with open(os.path.join(out_dir, 'samples.csv'), 'r', encoding='utf-8') as in_file:
            for expected, row in zip(expected_samples_log, in_file, strict=True):
                assert expected == row.strip(), "Samples mismatch"
        with open(os.path.join(out_dir, 'outcomes.csv'), 'r', encoding='utf-8') as in_file:
            outcome_rows = [row.strip() for row in in_file]
        assert outcome_rows[0] == 'evaluation_id,step,outcome_name,time,predicted_value'
        assert outcome_rows[1] == '"0:0",0,"outcome","2000-01-01T00:00:00",0'
        assert outcome_rows[-1] == '"3:0",3,"outcome","2000-01-01T00:00:00",-6.0'
        assert len(outcome_rows) == 12
        with open(os.path.join(out_dir, 'targets.csv'), 'r', encoding='utf-8') as in_file:
            target_rows = [row.strip() for row in in_file]
        assert target_rows == [
            'objective_name,outcome_name,time,target_value',
            '"neg_sum","outcome","2000-01-01T00:00:00",42'
        ]

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_out
    assert captured.err.strip() == expected_err

    assert result['a']['best'] == 1
    assert result['b']['best'] == 2
    assert result['c']['best'] == 3


def test_sequential_optimizer_initial_sim_fails(capsys):
    '''Test that sequential optimizer handles initial sim failing'''
    # pylint: disable=too-many-locals
    parameters = [
        CategoricalParameter('a', [0,1]),
        CategoricalParameter('b', [0,1,2]),
        CategoricalParameter('c', [0,1,2,3]),
    ]

    problem = MockProblem(parameters, Objective("neg_sum"), error={"sim" : MockError()})
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisySequentialOptimizer(problem, logger)
            with pytest.raises(RuntimeError, match="Initial simulation failed"):
                optimizer.optimize()

def test_sequential_optimizer_all_failing_after_initial():
    '''Test that sequential optimizer handles sims failing after initial'''
    # pylint: disable=too-many-locals
    parameters = [
        CategoricalParameter('a', [0,1]),
        CategoricalParameter('b', [0,1,2]),
        CategoricalParameter('c', [0,1,2,3]),
    ]

    problem = ProblemFailAfterN(1, parameters)
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisySequentialOptimizer(problem, logger)
            with pytest.raises(RuntimeError, match="All simulations failed"):
                optimizer.optimize()
