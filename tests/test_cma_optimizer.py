# pylint: disable=relative-beyond-top-level
import csv
import tempfile
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
        with open(f'{out_dir}/parameters.csv', 'r', encoding='utf-8') as in_file:
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

    for k,v in result.items():
        assert v['mean_transformed'] == approx(beale_function.amin[k])
