# pylint: disable=R0801
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from ax.api.client import Client
from .ax import daisy_param_to_ax_param
from .multi_objective import MultiObjective

@dataclass
class AxResult:
    '''Class holding Ax optimization result'''
    parameters : dict
    metrics : dict

class DaisyAxOptimizer:
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
        self.multi_objective = isinstance(problem.objective_fn, MultiObjective)
        if self.multi_objective:
            objective_str = ','.join([f'-{f.name}' for f in problem.objective_fn.objective_fns])
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
        num_trials = 0
        max_trials = self.options['max_trials']
        max_trials_iteration = self.options['max_trials_iteration']
        with ProcessPoolExecutor(self.number_of_processes) as executor:
            while num_trials < self.options['max_trials']:
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
                for i, result in enumerate(executor.map(self.problem, parameter_sets)):
                    log = { 'trial' : trial_indices[i] }
                    for name, value in named_parameter_sets[i].items():
                        log[f'param_{name}'] = value
                    for name, value in result.items():
                        log[f'metric_{name}'] = value
                    self.logger.result(**log)
                    self.client.complete_trial(trial_index=trial_indices[i], raw_data=result)
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
