import numpy as np
from daisypy.optim import DaisyOptimizationProblem, ContinuousParameter, MultiObjective
from .mockup import (MockRunner, MockFileGenerator, MockObjective)


def test_runner_succeds(tmp_path):
    '''Test that the return value is as expected when the runner succeds'''
    file_generator = MockFileGenerator({'dai' : ''})
    runner = MockRunner()
    parameters = { 'dai' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective('mock', 123)

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
    objective = MockObjective('mock', 123)

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_dir
    )
    result = problem([-1])
    assert np.isnan(result['mock'])

def test_multi_objective(tmp_path):
    '''Test that the return value is as expected for multiple objectives'''
    file_generator = MockFileGenerator({'dai' : ''})
    runner = MockRunner()
    parameters = { 'dai' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objectives = [ MockObjective(f'mock-{i}', i*123) for i in range(3) ]
    objective = MultiObjective('multi', objectives)

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_dir
    )
    result = problem([0])
    for obj in objectives:
        assert result[obj.name] == obj.value
