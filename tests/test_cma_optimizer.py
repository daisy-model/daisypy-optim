# pylint: disable=relative-beyond-top-level
import tempfile
from pytest import approx
from daisypy.optim import (
    DefaultLogger,    
    DaisyCMAOptimizer,
)
from .mock_problem import MockProblem
from .test_objectives import beale_function

def test_cma_optimizer():
    '''Test that CMA can optimize the Beale function'''
    problem = MockProblem(beale_function.parameters, beale_function)
    out_dir = 'out/cma'
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyCMAOptimizer(problem, logger, cma_options = { "maxfevals" : 500 })
            result = optimizer.optimize()
    for k,v in result.items():
        assert v['mean_transformed'] == approx(beale_function.amin[k])
