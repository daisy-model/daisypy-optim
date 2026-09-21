# pylint: disable=use-implicit-booleaness-not-comparison
from daisypy.optim import (
    DaisyOptimizationProblem,
    DaiFileGenerator,
    ContinuousParameter,
    MultiObjective,
    Simulation,
)
from daisypy.optim.process_executor import DaisyProcessExecutor
from .mockup import (MockRunner, MockFileGenerator, MockObjective)


def test_runner_succeeds(tmp_path):
    '''Test that the return value is as expected when the runner succeeds'''
    file_generators = { "runfile" : MockFileGenerator('') }
    output_specs = {}
    simulations = { "mock-sim" : Simulation(file_generators, output_specs) }
    outcome_specs = {}
    post_processing = {}
    runner = MockRunner()
    parameters = { 'runfile' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective('mock', 123)

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, out_dir
    )
    results_by_param_set_idx, errors = problem.evaluate([[-1]], DaisyProcessExecutor(1))
    assert len(results_by_param_set_idx) == 1
    assert 0 in results_by_param_set_idx
    assert errors == {}
    objectives, outcomes = results_by_param_set_idx[0]
    assert objectives['mock'] == objective.value
    assert outcomes == {}


def test_runner_fails(tmp_path):
    '''Test that the return value is nan when the runner fails'''
    file_generators = { "runfile" : MockFileGenerator('') }
    output_specs = {}
    simulations = { "mock-sim" : Simulation(file_generators, output_specs) }
    outcome_specs = {}
    post_processing = {}
    runner = MockRunner(returncode=1)
    parameters = { 'runfile' : [ContinuousParameter('p', 0, (-1, 1))] }
    out_dir = tmp_path
    objective = MockObjective('mock', 123)

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, out_dir
    )

    results, errors_by_param_set_idx = problem.evaluate([[-1]], DaisyProcessExecutor(1))
    assert results == {}
    assert len(errors_by_param_set_idx) == 1
    assert 0 in errors_by_param_set_idx
    errors = errors_by_param_set_idx[0]
    assert "mock-sim" in errors and errors["mock-sim"].returncode == 1

def test_multi_objective(tmp_path):
    '''Test that the return value is as expected for multiple objectives'''
    file_generators = { "runfile" : MockFileGenerator('') }
    output_specs = {}
    simulations = { "mock-sim" : Simulation(file_generators, output_specs) }
    outcome_specs = {}
    post_processing = {}
    runner = MockRunner()
    parameters = { 'runfile' : [ContinuousParameter('p', 0, (-1, 1))] }
    objectives = [ MockObjective(f'mock-{i}', i*123) for i in range(3) ]
    objective = MultiObjective('multi', objectives)

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, tmp_path
    )
    results_by_param_set_idx, errors = problem.evaluate([[0]], DaisyProcessExecutor(1))
    assert len(results_by_param_set_idx) == 1
    assert 0 in results_by_param_set_idx
    assert errors == {}
    objective_values = results_by_param_set_idx[0][0]
    for obj in objectives:
        assert objective_values[obj.name] == obj.value

def test_spawn_parallel_param_is_set(tmp_path):
    '''Test that the return value is as expected when the runner succeeds'''
    file_generators = {
        "runfile" : DaiFileGenerator(
            'runfile.dai',
            template_text='(defprogram p spawn (program p1 p2))'
        )
    }
    output_specs = {}
    simulations = { "mock-sim" : Simulation(file_generators, output_specs) }
    outcome_specs = {}
    post_processing = {}
    runner = MockRunner()
    parameters = { }
    out_dir = tmp_path
    objective = MockObjective('mock', 123)

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, out_dir
    )
    results_by_param_set_idx, errors = problem.evaluate([[]], DaisyProcessExecutor(1))
    assert len(results_by_param_set_idx) == 1
    assert 0 in results_by_param_set_idx
    assert errors == {}
    objective_value, outcomes = results_by_param_set_idx[0]
    assert objective_value['mock'] == objective.value
    assert outcomes == {}
