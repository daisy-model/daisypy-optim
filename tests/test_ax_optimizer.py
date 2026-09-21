# pylint: disable=relative-beyond-top-level,missing-function-docstring
import math
import tempfile
import pytest
from daisypy.optim import (
    DefaultLogger,
    ContinuousParameter
)
from daisypy.optim.ax_optimizer import DaisyAxOptimizer
from .mockup import MockProblem, MockObjective, MockError
from .test_objectives import beale_function

class LogScale:
    '''Ax needs objectives to be scaled better than the Beale function is by default'''
    def __init__(self, objective):
        self._objective = objective

    @property
    def name(self):
        return self._objective.name

    def __call__(self, *args, **kwargs):
        return { k : math.log(v) for k,v in self._objective(*args, **kwargs).items() }

def test_ax_optimizer_all_fails():
    '''Test that ax throws when all simulations fail'''
    parameters = [
        ContinuousParameter('a', 0, (-1, 1)),
        ContinuousParameter('b', 0, (-1, 1)),
        ContinuousParameter('c', 0, (-1, 1)),
    ]

    problem = MockProblem(parameters, MockObjective(), error={'sim': MockError()})
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyAxOptimizer(
                problem, logger, { 'max_trials' : 10, 'max_trials_iteration' : 3 }
            )
            with pytest.raises(RuntimeError, match='All simulations failed'):
                optimizer.optimize()

def test_ax_optimizer_all_nans():
    '''Test that ax throws when all simulations fail'''
    parameters = [
        ContinuousParameter('a', 0, (-1, 1)),
        ContinuousParameter('b', 0, (-1, 1)),
        ContinuousParameter('c', 0, (-1, 1)),
    ]

    problem = MockProblem(parameters, MockObjective(value=math.nan))
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyAxOptimizer(
                problem, logger, { 'max_trials' : 10, 'max_trials_iteration' : 3 }
            )
            with pytest.raises(RuntimeError, match='All simulations failed'):
                optimizer.optimize()


def test_mixed_params_type():
    '''Test that ax throws when all simulations fail'''
    parameters = beale_function.parameters.copy()
    parameters[0] = parameters[0].as_categorical(10)
    problem = MockProblem(parameters, LogScale(beale_function))
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyAxOptimizer(
                problem, logger, { 'max_trials' : 5, 'max_trials_iteration' : 5 }
            )
            optimizer.optimize()
    # Nothing to assert, we are just checking that things run ...

@pytest.mark.slow
def test_ax_optimizer():
    '''Test that Ax can optimize the Beale function'''
    problem = MockProblem(beale_function.parameters, LogScale(beale_function))
    with tempfile.TemporaryDirectory() as out_dir:
        with DefaultLogger(out_dir) as logger:
            optimizer = DaisyAxOptimizer(
                problem, logger, { 'max_trials' : 200, 'max_trials_iteration' : 5 }
            )
            result = optimizer.optimize()

    for k, v in result.parameters.items():
        assert v == pytest.approx(beale_function.amin[k], abs=0.1)
