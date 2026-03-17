import numpy as np
from daisypy.optim import DaisyOptimizationProblem, ContinuousParameter
from .mockup import (MockRunner, MockFileGenerator, MockObjective)


def test_runner_succeds(tmp_path):
    '''Test that the return value is as expected when the runner succeds'''
    file_generator = MockFileGenerator({'dai' : ''})
    runner = MockRunner()
    parameters = { 'dai' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective(123)

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_dir
    )
    result = problem([-1])
    assert result['mock'] == objective.value


def test_runner_fails(tmp_path):
    '''Test that the return value is nan when the runner fails'''
    file_generator = MockFileGenerator({'dai' : ''})
    runner = MockRunner(returncode=1)
    parameters = { 'dai' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective(123)

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_dir
    )
    result = problem([-1])
    assert np.isnan(result['mock'])
