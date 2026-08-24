from daisypy.optim import (
    DaisyOptimizationProblem,
    ContinuousParameter,
    MultiObjective,
    Simulation,
)
from .mockup import (MockRunner, MockFileGenerator, MockObjective)


def test_runner_succeds(tmp_path):
    '''Test that the return value is as expected when the runner succeds'''
    file_generators = { "dai" : MockFileGenerator('') }
    outcome_specs = {}
    simulations = { "mock-sim" : Simulation(file_generators, outcome_specs) }
    post_processing = {}
    runner = MockRunner()
    parameters = { 'dai' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective('mock', 123)

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, out_dir
    )
    objective_value, outcomes, errors = problem([-1])
    assert objective_value['mock'] == objective.value
    assert outcomes == {}
    assert errors == {}


def test_runner_fails(tmp_path):
    '''Test that the return value is nan when the runner fails'''
    file_generators = { "dai" : MockFileGenerator('') }
    outcome_specs = {}
    simulations = { "mock-sim" : Simulation(file_generators, outcome_specs) }
    post_processing = {}
    runner = MockRunner(returncode=1)
    parameters = { 'dai' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective('mock', 123)

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, out_dir
    )

    errors = problem([-1])[2]
    assert "mock-sim" in errors and errors["mock-sim"].returncode == 1

def test_multi_objective(tmp_path):
    '''Test that the return value is as expected for multiple objectives'''
    file_generator = { "dai" : MockFileGenerator('') }
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
