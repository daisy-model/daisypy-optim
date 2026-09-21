import math
import tempfile
import os
import platform
from pathlib import Path
import concurrent
from daisypy.optim.output_store import OutputStore

class DaisyOptimizationProblem:
    # pylint: disable=too-many-instance-attributes
    '''A DaisyOptimizationProblem maps parameters to objectives, and is defined by
      - a list of simulations that should be "run as one"
      - a runner that knows how to run Daisy
      - a set of parameters
      - an objective
    '''
    def __init__(self,
                 runner,
                 simulations,
                 outcome_specs,
                 post_processing,
                 objective_fn,
                 parameters,
                 data_dir=None,
                 debug=False,
                 outcome_filters=None):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Parameters
        ----------
        runner : DaisyRunner

        simulations : dict of (str, Simulation)
          A dict of named simulations to run using the same set of parameters

        outcome_specs : {str : (str, str, str)}
          A dict defining with named outcomes. Each outcome is defined by a triplet of strings,
          (simulation, outcome, variable) that uniquely identifies a column in a outcome file

        objective_fn : daisypy.optim.Objective
          An objective function that computes one or more named objective values from a dict of
          named DataFrames.

        parameters : dict of (str, [DaisyParameter])
          Parameters to optimize. Mapping from parameter groups, e.g. "runfile" to specifications

        data_dir : str
          If not None then temporary directories will be created in this directory. Otherwise, they
          will be created in a default location depending on platform.

        debug: bool
          If True do not delete the temporary directory where Daisy output is stored

        outcome_filters : [str] or None
          List of outcomes to keep. If None all outcomes are kept. This is useful to avoid logging
          intermediate outcomes that have no interest.
        """
        self.runner = runner
        self.simulations = simulations
        self.outcome_specs = outcome_specs
        self.post_processing = post_processing
        self.objective_fn = objective_fn
        self.parameter_kind = {}

        # Convert dict of parameters to a list of parameters and verify that there are no name
        # clashes.
        self.parameters = []
        for kind, params in parameters.items():
            for param in params:
                if param.name in self.parameter_kind:
                    raise ValueError(
                        'Parameters must have unique names across all templates. '
                        f'{param.name} from {kind} is also in {self.parameter_kind[param.name]}'
                    )
                self.parameter_kind[param.name] = kind
                self.parameters.append(param)

        self.data_dir = data_dir
        if data_dir is None and platform.system().lower() == 'linux':
            # There is a good chance that we are using flatpak, in which case we need to use a tmp
            # location that flatpak Daisy can read and write. Anything inside the user home is good.
            self.data_dir = os.path.expanduser('~/.tmp/daisy')
        if self.data_dir is not None:
            os.makedirs(self.data_dir, exist_ok=True)
        self.debug = debug
        self.outcome_filters = outcome_filters

    def process_demand(self, max_processes):
        '''Compute the combined process demand for all simulations in this problem. If the raw total
        demand increases max processes, then it is adjusted such that it fits. This adjustment is
        done per simulation to minimize expected overall runtime while also avoiding demanding
        processes that are not needed to achieve that expected runtime.

        Parameters
        ----------
        max_processes : int > 0
          Soft cap on the process demand. The demand is at least equal to the number of simulations
          in the problem.

        Returns
        -------
        total_process_demand : int >= 0
        '''
        total_demand = 0
        for _, demand in self._process_budget(max_processes).values():
            total_demand += demand
        return total_demand

    def evaluate(self, parameter_sets, executor):
        '''Evaluate the problem on a list of parameter sets using a given executor

        Parameters
        ----------
        parameter_sets : [[float]]
          List of list of parameter values. The outer list groups parameters in sets that are
          evaluated together. The inner lists contain all parameters for one problem evaluation and
          MUST match `self.parameters` such that parameter_sets[i][j] is the i'th sample of
          self.parameters[j]

        executor : DaisyProcessExecutor

        Returns
        -------
        results, errors.
          results is { param_set_idx : ( objective, outcomes ) }
          errors is { param_set_idx : { sim_name : error } }
        '''
        named_parameter_sets = [self.wrap_parameters(param_set) for param_set in parameter_sets]
        with tempfile.TemporaryDirectory(dir=self.data_dir, delete=not self.debug) as base_dir:
            return self._run(base_dir, named_parameter_sets, executor)

    def wrap_parameters(self, parameter_values):
        '''Map parameter values to names

        Parameters
        ----------
        parameter_values : sequence
          Parameter values. Length MUST match length of ``self.parameters``.

        Returns
        -------
        named_parameters : { str : { str : float }}
          Keys in outer dict are names of file generators. Keys in inner dict are parameter names
        '''
        named_parameters = { 'runfile' : {} }
        for p, value in zip(self.parameters, parameter_values):
            kind = self.parameter_kind[p.name]
            if kind not in named_parameters:
                named_parameters[kind] = { p.name : value }
            else:
                named_parameters[kind][p.name] = value
        return named_parameters

    def _prepare(self, base_dir, named_parameter_sets, process_budget):
        # Setup all simulations and return them in a dict mapping time cost to process cost to sim
        base_dir = Path(base_dir)
        time_map = {}
        # We want the time map to be sorted such that
        #  - the first outer key is the largest time value, and the last outer key is the smallest
        #    time value.
        #  - The first inner key is the largest cost value, and the last inner key is the smallest
        #    cost value.
        # This works because iter(dict) maintains insertion order.
        for time, cost in sorted(process_budget.values(), reverse=True):
            if not time in time_map:
                time_map[time] = { cost : [] }
            elif not cost in time_map[time]:
                time_map[time][cost] = []

        for i, named_parameter_set in enumerate(named_parameter_sets):
            for name, sim in self.simulations.items():
                sim_dir = base_dir / f'param-set-{i}' / name
                sim_dir.mkdir(parents=True)
                time, cost = process_budget[name]
                sim_file, outputs = sim.setup(sim_dir, named_parameter_set, spawn_parallelism=cost)
                time_map[time][cost].append((
                    cost, i, name, outputs, { 'dai_file' : sim_file, 'output_directory' : sim_dir }
                ))

        return time_map


    def _process_budget(self, max_processes):
        # Adjust process budget such that no simulation requests more processes than are available
        budget = {}
        for sim_name, sim in self.simulations.items():
            if sim.process_cost > max_processes:
                budget[sim_name] = _find_best_budget(sim.process_cost, max_processes)
            else:
                # Time cost is 1 unit when full process_cost is allocated
                budget[sim_name] = (1, sim.process_cost)
        return budget


    def _run(self, base_dir, named_parameters, executor):
        # pylint: disable=too-many-locals
        '''Run Daisy in ``base_dir`` and evaluate the objective on the produced files.'''
        # We need to run all simulations in the problem before we can evaluate the objective
        budget = self._process_budget(executor.max_processes)
        # Maybe we should do something about the order we submit processes in?
        time_map = self._prepare(base_dir, named_parameters, budget)

        max_processes = executor.max_processes
        allocated = 0
        running = set()
        run_results = []
        # Now we need to schedule things
        while len(time_map) > 0:
            # Allocate sims in order from most expensive (longest running, most processes) to least
            # expensive (shortest running, fewest processes)
            remaining = {}
            for time, cost_map in time_map.items():
                for cost, sim_params in cost_map.items():
                    while allocated + cost <= max_processes and len(sim_params) > 0:
                        params = sim_params.pop()
                        running.add(executor.submit(self._run_one, *params))
                        allocated += cost
                    # If there are still simulations to run for this time/cost combination, then we
                    # add it to the remaining time_map
                    if len(sim_params) > 0:
                        if time not in remaining:
                            remaining[time] = { cost : sim_params }
                        else:
                            remaining[time][cost] = sim_params
            time_map = remaining

            if len(time_map) == 0:
                # All sims have been scheduled, now we just wait
                wait_on = concurrent.futures.ALL_COMPLETED
            else:
                wait_on = concurrent.futures.FIRST_COMPLETED
            done, running = concurrent.futures.wait(running, return_when=wait_on)
            for future in done:
                run_result = future.result()
                allocated -= run_result[0]
                run_results.append(run_result[1:])

        assert len(running) == 0, 'List of running processes is not empty'
        assert allocated == 0, 'Process budget calculation mismatch'

        errors, outputs = self._gather_errors_and_outputs(run_results)
        results = self._gather_results(errors, outputs)

        return results, errors

    def _run_one(self, cost, idx, sim_name, outputs, params):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        sim_result = self.runner(**params)
        return cost, idx, sim_name, outputs, sim_result

    def _gather_errors_and_outputs(self, run_results):
        errors = {}
        outputs = {}
        for run_result in run_results:
            param_set_idx, sim_name, sim_outputs, sim_result = run_result
            if sim_result.returncode != 0:
                if param_set_idx not in errors:
                    errors[param_set_idx] = {}
                errors[param_set_idx][sim_name] = sim_result
            else:
                if param_set_idx not in outputs:
                    outputs[param_set_idx] = {}
                outputs[param_set_idx][sim_name] = sim_outputs
        return errors, outputs

    def _gather_results(self, errors, outputs):
        results = {}
        for param_set_idx, param_set_outputs in outputs.items():
            if param_set_idx in errors:
                # We only gather outcomes and compute objectives for parameter sets where all
                # simulations succeed.
                continue
            output_store = OutputStore(param_set_outputs)
            outcomes = {
                name : output_store.extract(*spec) for name, spec in self.outcome_specs.items()
            }
            for name, p in self.post_processing.items():
                outcomes[name] = p(outcomes)
            if self.outcome_filters is not None:
                outcomes = { k : outcomes[k] for k in self.outcome_filters }
            results[param_set_idx] = (self.objective_fn(outcomes), outcomes)
        return results

def _find_best_budget(requested, available):
    # This will find the smallest of the fastest feasible allocations
    # For example, requested 8, available 7 returns 4, because 4 processes will finish the task in
    # two rounds of processing. Using fewer leads to more rounds of processing, using more will not
    # lead to fewer rounds of processing
    alloc = []
    for i in range(1, available+1):
        alloc.append((math.ceil(requested/i), i))
    alloc = sorted(alloc)
    return alloc[0] # time, allocated
