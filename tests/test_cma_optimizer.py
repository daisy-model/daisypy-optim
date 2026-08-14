# pylint: disable=relative-beyond-top-level
import os
import csv
import tempfile
from types import SimpleNamespace
import numpy as np
from pytest import approx
from daisypy.optim import (
    DefaultLogger,
    DaisyCMAOptimizer,
)
from .mockup import MockProblem
from .test_objectives import beale_function

def test_cma_optimizer():
    '''Test that CMA can optimize the Beale function'''
    problem = MockProblem(beale_function.parameters, beale_function)
    out_dir = 'out/cma'
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyCMAOptimizer(problem, logger, cma_options = { "maxfevals" : 500 })
            result = optimizer.optimize()
        params_path = os.path.join(out_dir, 'parameters.csv')
        with open(params_path, 'r', encoding='utf-8', newline='') as in_file:
            rows = list(csv.DictReader(in_file))

    assert rows
    assert 'param_x_mean' in rows[0]
    assert 'param_y_mean' in rows[0]
    assert 'param_x__param_x_cov' in rows[0]
    assert 'param_x__param_y_cov' in rows[0]
    assert 'param_y__param_y_cov' in rows[0]
    assert 'param_y__param_x_cov' not in rows[0]
    assert 'param_x_std' not in rows[0]
    assert 'param_y_std' not in rows[0]
    assert {row['tag'] for row in rows} == {'raw', 'standardized'}
    assert {row['distribution'] for row in rows} == {'multivariate_normal'}

    expected_covariance = optimizer.optimizer.sigma**2 * (
        optimizer.optimizer.sigma_vec.transform_covariance_matrix(
            optimizer.optimizer.sm.C.copy()
        )
    )
    last_standardized_row = [row for row in rows if row['tag'] == 'standardized'][-1]
    assert float(last_standardized_row['param_x__param_x_cov']) == approx(expected_covariance[0, 0])
    assert float(last_standardized_row['param_x__param_y_cov']) == approx(expected_covariance[0, 1])
    assert float(last_standardized_row['param_y__param_y_cov']) == approx(expected_covariance[1, 1])

    raw_scaling = np.outer(optimizer.objective.multiplier, optimizer.objective.multiplier)
    expected_raw_covariance = raw_scaling * expected_covariance
    last_raw_row = [row for row in rows if row['tag'] == 'raw'][-1]
    assert float(last_raw_row['param_x__param_x_cov']) == approx(expected_raw_covariance[0, 0])
    assert float(last_raw_row['param_x__param_y_cov']) == approx(expected_raw_covariance[0, 1])
    assert float(last_raw_row['param_y__param_y_cov']) == approx(expected_raw_covariance[1, 1])

    for k,v in result.items():
        assert v['mean_transformed'] == approx(beale_function.amin[k])

def test_cma_optimizer_logs_termination_criteria(capsys):
    '''Test that CMA logs configured and final termination criteria'''
    problem = MockProblem(beale_function.parameters, beale_function)
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyCMAOptimizer(
                problem,
                logger,
                cma_options = { "maxfevals" : 30, "verbose" : -9 }
            )
            optimizer.optimize()

    captured = capsys.readouterr()
    assert 'Configured termination criteria' in captured.out
    assert 'termination_criterion=maxfevals,threshold=30' in captured.out
    assert 'termination_criterion=tolx,threshold=1e-11' in captured.out
    assert 'Termination criteria status' in captured.out
    assert 'termination_criterion=maxfevals,threshold=30,current_value=' in captured.out
    assert 'triggered=True' in captured.out

def test_cma_optimizer_reports_tolstagnation_status():
    '''Test that tolstagnation status is reported from CMA history'''
    optimizer = DaisyCMAOptimizer.__new__(DaisyCMAOptimizer)
    optimizer.optimizer = SimpleNamespace(
        opts={'tolstagnation' : 20},
        fit=SimpleNamespace(
            histbest=np.array([1.0, 2.0, 3.0, 4.0]),
            histmedian=np.array([2.0, 4.0, 6.0, 8.0]),
        ),
        countiter=10,
    )
    # pylint: disable=protected-access
    result = optimizer._termination_criterion_value('tolstagnation')

    assert result == {
        'window' : 2,
        'median_history_previous' : 3.0,
        'median_history_recent' : 7.0,
        'best_history_previous' : 1.5,
        'best_history_recent' : 3.5,
    }
