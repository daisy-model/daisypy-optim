# pylint: disable=R0801
# TODO: This needs a lot more work
#       * Parameter scaling
#       * Documentation regarding objective scaling
#       * Termination criteria
#          - Convergence in objective value
#          - Convergence in sampling distribution
import logging
import math
from dataclasses import dataclass
from ax.api.client import Client
from daisypy.optim.ax import daisy_param_to_ax_param
from daisypy.optim.multi_objective import MultiObjective
from daisypy.optim.outcome_logging import log_outcomes
from daisypy.optim.target_logging import log_targets
from daisypy.optim.process_executor import DaisyProcessExecutor

# Silence Ax info logging
logging.disable(logging.INFO)

@dataclass
class AxResult:
    '''Class holding Ax optimization result'''
    parameters : dict
    metrics : dict

class DaisyAxOptimizer:
    # pylint: disable=too-few-public-methods,too-many-locals
    """Daisy optimizer using Ax. Can do scalar and multi objective optimization"""
    def __init__(self, problem, logger, options=None, number_of_processes=None):
        """
        Parameters
        ----------
        problem : DaisyProblem

        options : dict
        """
        self.problem = problem
        self.logger = logger
        self.number_of_processes = number_of_processes

        if options is None:
            options = {}
        options.setdefault('max_trials', 500)
        options.setdefault('max_trials_iteration', 3)
        options.setdefault('stagnation_window_length', 10)
        options.setdefault('stagnation_relative_tolerance', 1e-3)
        options.setdefault('stagnation_absolute_tolerance', 1e-6)
        self.options = options

        ax_parameters = [ daisy_param_to_ax_param(p) for p in self.problem.parameters ]

        self.client = Client()
        self.client.configure_experiment(parameters=ax_parameters)

        # TODO: Assumes we minimize
        self.multi_objective = (isinstance(problem.objective_fn, MultiObjective) and
                                problem.objective_fn.aggregate_fn is None)
        if self.multi_objective:
            objective_str = ','.join([f'-{f.name}' for f in problem.objective_fn.objectives])
        else:
            objective_str = f'-{problem.objective_fn.name}'
        self.client.configure_optimization(objective=objective_str)


    def optimize(self):
        # pylint: disable=too-many-branches,too-many-statements
        '''Run the optimizer and return the result. The result is a single AxResult when doing
        scalar optimization and a list of AxResult when doing multi optimization

        Returns
        -------
        { 'sample' : AxResult, 'pred' : AxResult } OR a list with a dict for each objective
        '''
        # TODO: Log parameter distributions
        num_trials = 0
        num_failed_trials = 0
        max_trials = self.options['max_trials']
        max_trials_iteration = self.options['max_trials_iteration']
        stagnation_window = self.options['stagnation_window_length'] # In steps
        stagnation_rtol = self.options['stagnation_relative_tolerance']
        stagnation_atol = self.options['stagnation_absolute_tolerance']
        best_objective_values = {}
        step = 0
        log_targets(self.logger, self.problem.objective_fn)
        self.logger.persist()
        with DaisyProcessExecutor(self.number_of_processes) as executor:
            # Decide how many trials to generate in each step
            # If possible we want trials to run perfectly parallel. This means that the number
            # trials need to be capped such that
            #   processes_used_per_evaluations * num_trials <= max_processes
            # At the same time we must have at least 1 trial and we also want to respect the user
            # supplied `max_trials_iteration` option.
            process_demand = self.problem.process_demand(executor.max_processes)
            num_trials_iteration = min(
                max_trials_iteration, max(1, executor.max_processes // process_demand)
            )
            while num_trials < self.options['max_trials']:
                if _stagnated(
                        best_objective_values, stagnation_window, stagnation_rtol, stagnation_atol
                ):
                    self.logger.info('Optimization done: Objective value converged')
                    break
                step += 1
                max_trials_this_iteration = min(num_trials_iteration, max_trials - num_trials)
                trials = self.client.get_next_trials(max_trials=max_trials_this_iteration)
                trial_indices = []
                parameter_sets = []
                named_parameter_sets = []
                for trial_index, sampled_parameters in trials.items():
                    trial_indices.append(trial_index)
                    named_params = {
                        p.name : sampled_parameters[p.name] for p in self.problem.parameters
                    }
                    named_parameter_sets.append(named_params)
                    params = [sampled_parameters[p.name] for p in self.problem.parameters]
                    parameter_sets.append(params)

                results, errors = self.problem.evaluate(parameter_sets, executor)
                num_failed_trials += len(errors)

                best_trial = {}
                for sample_idx, result in results.items():
                    trial_idx = trial_indices[sample_idx]
                    all_objectives_finite = True
                    for name, value in result[0].items():
                        if not math.isfinite(value):
                            all_objectives_finite = False
                            self.logger.warning(
                                step=step, sample_idx=sample_idx,
                                msg="Non finite objective value"
                            )
                            self.client.mark_trial_failed(trial_index=trial_idx)
                            num_failed_trials += 1
                        else:
                            if name not in best_trial:
                                best_trial[name] = value
                            else:
                                best_trial[name] = min(value, best_trial[name])
                    if all_objectives_finite:
                        param_set = named_parameter_sets[sample_idx]
                        self._log_result(step, sample_idx, trial_idx, param_set, result)
                        self.client.complete_trial(trial_index=trial_idx, raw_data=result[0])
                for sample_idx, error in errors.items():
                    trial_idx = trial_indices[sample_idx]
                    self._log_error(step, sample_idx, trial_idx, error)
                    self.client.mark_trial_failed(trial_index=trial_idx)

                for k,v in best_trial.items():
                    self.logger.info(step=step,metric=k,best_trial_value=v)
                    if not k in best_objective_values:
                        best_objective_values[k] = []
                    best_objective_values[k].append(v)
                num_trials += len(trials)
                self.logger.persist()

        if num_failed_trials == num_trials:
            self.logger.error(f'All {num_trials} trials failed')
            self.logger.persist()
            raise RuntimeError('All simulations failed')

        if self.multi_objective:
            # Handle multi objective result
            result = {
                'pred' : [
                    AxResult(parameters, metrics)
                    for parameters, metrics, _, _ in self.client.get_pareto_frontier(True)
                ],
                'sample' : [
                    AxResult(parameters, metrics)
                    for parameters, metrics, _, _ in self.client.get_pareto_frontier(False)
                ],
            }
        else:
            # Handle scalar objective result
            result = {
                'pred' : AxResult(*self.client.get_best_parameterization(True)[:2]),
                'sample' : AxResult(*self.client.get_best_parameterization(False)[:2]),
            }
        return result

    def _log_result(self, step, sample_idx, trial_idx, param_set, result):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        params = {
            f'param_{name}' : value  for name, value in param_set.items()
        }
        objective, outcome = result
        objective = { f'metric_{k}' : v for k,v in objective.items() }
        self.logger.samples(
            step=step,
            index=sample_idx,
            tag="raw",
            trial=trial_idx,
            **objective,
            **params
        )
        log_outcomes(
            self.logger,
            outcome,
            step=step,
            index=sample_idx,
            trial=trial_idx,
        )

    def _log_error(self, step, sample_idx, trial_idx, error):
        # error is { sim_name : CompletedProcess }
        for name, e in error.items():
            self.logger.warning(
                step=step,
                index=sample_idx,
                trial_idx=trial_idx,
                sim_name=name,
                msg=f"Simulation failed with exit code '{e.returncode}'"
            )

def _stagnated(history, window, rtol, atol):
    eps = 1e-12
    if len(history) == 0:
        return False
    for v in history.values():
        if len(v) < window + 1:
            return False
        for i in range(window):
            prev = v[-(window+1) + i]
            curr = v[-(window+1) + i + 1]
            delta = abs(prev - curr)
            denom = max(abs(prev), abs(curr), eps)
            if delta/denom > rtol or delta > atol:
                return False
    return True
