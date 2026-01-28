# pylint: disable=relative-beyond-top-level
import tempfile
import pytest
from pytest import approx
from daisypy.optim import (
    DefaultLogger,
    DaisySkoptOptimizer,
)
from .mock_problem import MockProblem
from .test_objectives import beale_function

@pytest.mark.slow
def test_skopt_optimizer():
    '''Test that Skopt can optimize the Beale function. It is very slow and wont actually find
    the optimum because maxfevals is too low. But it shows that it "works".'''
    problem = MockProblem(beale_function.parameters, beale_function)
    out_dir = 'out/cma'
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisySkoptOptimizer(problem, logger, options = { "maxfevals" : 50 })
            result = optimizer.optimize()
    for k,v in result.items():
        assert v['best'] == approx(beale_function.amin[k])
