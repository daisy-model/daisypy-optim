# pylint: disable=R0801
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from ax.api.client import Client
from daisypy.optim.ax import daisy_param_to_ax_param
from daisypy.optim.multi_objective import MultiObjective
from daisypy.optim.outcome_logging import log_outcomes
from daisypy.optim.target_logging import log_targets
from daisypy.optim.util import get_single_scalar

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
        if number_of_processes is None:
            self.number_of_processes = multiprocessing.cpu_count()
        else:
            self.number_of_processes = number_of_processes

        if options is None:
            options = {}
        options.setdefault('max_trials', 10)
        options.setdefault('max_trials_iteration', 3)
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
        '''Run the optimizer and return the result. The result is a single AxResult when doing
        scalar optimization and a list of AxResult when doing multi optimization

        Returns
        -------
        AxResult OR list of AxResult
        '''
        # TODO: Log parameter distributions
        num_trials = 0
        max_trials = self.options['max_trials']
        max_trials_iteration = self.options['max_trials_iteration']
        step = 0
        log_targets(self.logger, self.problem.objective_fn)
        with ProcessPoolExecutor(self.number_of_processes) as executor:
            while num_trials < self.options['max_trials']:
                step += 1
                max_trials_this_iteration = min(max_trials_iteration, max_trials - num_trials)
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

                # Run simulations in parallel
                for i, (objective, outcomes, errors) in enumerate(
                        executor.map(self.problem, parameter_sets)):
                    trial_index = trial_indices[i]
                    if len(errors) > 0:
                        for sim, error in errors.items():
                            self.logger.warning(
                                step=step,
                                trial=trial_index,
                                msg=f"Simulation '{sim}' failed with exit code {error.returncode}"
                            )
                        self.client.mark_trial_failed(trial_index=trial_index)
                        continue

                    if not self.multi_objective:
                        # Verify that we have a single scalar objective
                        _ = get_single_scalar(objective)

                    log = {
                        'step' : step,
                        'index' : i,
                        'tag' : 'raw',
                        'trial' : trial_index
                    }
                    for name, value in named_parameter_sets[i].items():
                        log[f'param_{name}'] = value
                    for name, value in objective.items():
                        log[f'metric_{name}'] = value
                    self.logger.samples(**log)
                    log_outcomes(
                        self.logger,
                        outcomes,
                        step=step,
                        index=i,
                        trial=trial_index,
                    )
                    self.client.complete_trial(trial_index=trial_index, raw_data=objective)
                num_trials += len(trials)

        if self.multi_objective:
            # Handle multi objective result
            result = [
                AxResult(parameters, metrics)
                for parameters, metrics, _, _ in self.client.get_pareto_frontier()
            ]
        else:
            # Handle scalar objective result
            parameters, metrics, _, _ = self.client.get_best_parameterization()
            result = AxResult(parameters, metrics)
        return result
