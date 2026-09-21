# pylint: disable=too-few-public-methods,R0801
import numpy as np
from daisypy.optim.outcome_logging import log_outcomes
from daisypy.optim.target_logging import log_targets
from daisypy.optim.util import get_single_scalar
from daisypy.optim.process_executor import DaisyProcessExecutor

class DaisySequentialOptimizer:
    """Daisy optimizer using a sequential approach

    The method starts from the initial parameters. Then it changes each parameter in turn.
    The single parameter leading to best performance is then fixed and the process repeated
    untill all parameters are fixed.
    """
    def __init__(self, problem, logger, options=None, number_of_processes=None):
        """
        Parameters
        ----------
        problem : DaisyProblem

        logger : ...

        options : dict

        number_of_processes: int > 0 (Optional)
          The maximum number of processes to use when running Daisy. Defaults to
          os.process_cpu_count()
        """
        if len(problem.parameters) == 0:
            raise ValueError('Optimization problem has no parameters to optimize')

        if options is None:
            options = {}
        self.problem = problem
        self.logger = logger
        self.number_of_processes = number_of_processes

        # Standardize parameters such that they are all categorical and the initial value is the
        # first value in the values list.
        num_samples = options.get("num_samples", 3)
        self.parameters = []
        for param in problem.parameters:
            if param.type == 'Continuous':
                self.parameters.append(param.as_categorical(num_samples))
            elif param.type == 'Categorical':
                self.parameters.append(param.normal_form())
            else:
                raise ValueError(f'Unknown parameter type {param.type}')

    def optimize(self):
        '''Run optimization'''
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        # Recall that we are working with categorical parameters, so there is no sampling of new
        # parameters.
        total_f_evals = 0
        current_fval = np.inf
        step = 0
        fixed = set()   # The parameters that are already fixed
        floating = {}   # The parameters that we need to fix
        current = {}    # The parameter values that we are currently using
        tried = set()   # The parameter value combinations that have been tried
        order = []      # Order that parameters are passed to the problem.
        num_param_values = [] # Number of possible parameter values for each parameter
        for param in self.parameters:
            floating[param.name] = param.values
            current[param.name] = param.values[0] # Parameter values are tried in order
            order.append(param.name)
            num_param_values.append(len(param.values))

        min_evals, max_evals = _count_min_max_param_evals(num_param_values)
        self.logger.info(f'Using at least {min_evals} and at most {max_evals} function evaluations')
        log_targets(self.logger, self.problem.objective_fn)

        with DaisyProcessExecutor(self.number_of_processes) as executor:
            while len(floating) > 0:
                step += 1

                # Log the parameter distribution
                params = {
                    f'param_{name}_choices' : ','.join([str(v) for v in values])
                    for name, values in floating.items()
                }
                for name in fixed:
                    params[f'param_{name}_choices'] = str(current[name])
                self.logger.parameters(distribution='categorical', tag='raw', step=step, **params)

                param_sets, param_sets_ids = _generate_parameter_sets(
                    floating, current, order, tried
                )
                if step == 1:
                    # We need to add the initial parameters because they are skipped by the
                    # generator. None is used to signal that this parameter set is special and we
                    # should stop if it yields the best objective.
                    param_sets.append(tuple((current[name] for name in order)))
                    param_sets_ids.append((None, 0))
                self.logger.info(step=step, n_param_sets=len(param_sets))
                best = np.inf
                best_idx = None
                num_failures = 0
                results, errors = self.problem.evaluate(param_sets, executor)

                for param_set in param_sets:
                    tried.add(param_set)

                # If all parameter sets fail we give up
                if len(results) == 0:
                    self.logger.error('All parameter sets failed. Aborting')
                    self.logger.persist()
                    raise RuntimeError('All parameter sets failed')

                # Log all the errors
                for idx, sim_errors in errors.items():
                    param_id = param_sets_ids[idx]
                    num_failures += 1
                    for name, e in sim_errors.items():
                        self.logger.warning(
                            step=step,
                            msg=f"Simulation '{name}' ({idx}->{param_id}) failed with exit code "
                            f'{e.returncode}'
                        )

                # Log all the param sets that worked and find the best one
                for param_set_idx, (objective, outcomes) in results.items():
                    # We must test what happens when all fails
                    fval = get_single_scalar(objective)
                    objective_value = { f'metric_{k}' : v for k,v in objective.items() }
                    params = {
                        f'param_{name}' : value
                        for name, value in zip(order, param_sets[param_set_idx])
                    }
                    self.logger.samples(
                        step=step,
                        index=param_set_idx,
                        tag="raw",
                        **objective_value,
                        **params
                    )
                    log_outcomes(
                        self.logger, outcomes, step=step, index=param_set_idx
                    )
                    if np.isfinite(fval) and fval < best:
                        best = fval
                        best_idx = param_set_idx

                if best_idx is None:
                    self.logger.error(
                        'All successful simulations had non-finite objective values. Aborting'
                    )
                    self.logger.persist()
                    raise RuntimeError('All successful simulations had non-finite objective values')

                total_f_evals += len(param_sets)
                self.logger.info(step=step, total_function_evaluations=total_f_evals)
                if num_failures > 0:
                    self.logger.warning(step=step, n_failed_runs=num_failures)
                self.logger.info(step=step, best_objective=best)

                if best > current_fval:
                    # Nothing is better than using current values of all parameters, so we stop.
                    # We could consider setting a random parameter to a random value, or something
                    # similar.
                    self.logger.info('No improvement in objective. Stopping.')
                    self.logger.persist()
                    break

                name, idx = param_sets_ids[best_idx]
                if name is None:
                    self.logger.info(
                        'No improvement in objective over initial parameters. Stopping.'
                    )
                    self.logger.persist()
                    break
                current_fval = best
                value = floating.pop(name)[idx]
                current[name] = value
                fixed.add(name)
                self.logger.info(f'step={step},Fixing {name} to {value}')
                self.logger.persist()

        result = {}
        for k,v in current.items():
            result[k] = { 'best': v }
        return result

def _count_min_max_param_evals(num_param_values):
    # Count the minimum and maximum number of function evaluations
    # Worst case is that we always fix the parameter with fewest values
    num_param_values = sorted(num_param_values)
    max_evals = 1 # We always do one with the current parameter set
    for start in range(len(num_param_values)):
        for n in num_param_values[start:]:
            max_evals += n-1 #

    # Best case is that we always fix the parameter with most values
    min_evals = 1
    num_param_values = num_param_values[::-1]
    for n in num_param_values:
        min_evals += n-1
    return min_evals, max_evals

def _generate_parameter_sets(floating, current, order, tried):
    # Generate parameter sets where all parameters, exept one, are fixed
    # A parameter set is a dict of (parameter name, parameter value)
    # For a specific parameter p, we keep all other parameters fixed and then generate
    # parameter sets where p is varied over all its possible values
    param_sets_ids = []
    param_sets = []
    for name, values in floating.items():
        # We keep all except name fixed
        for i, value in enumerate(values):
            if value == current[name]:
                # This is the case where all parameters have their current value and we
                # have already computed this combination in the previous step regardless
                # of which parameter was fixed
                continue

            param_set = []
            for param_name in order: # We must maintain the order of parameters
                if param_name == name:
                    param_set.append(value)
                else:
                    param_set.append(current[param_name])
            param_set = tuple(param_set)
            if not param_set in tried:
                param_sets.append(param_set)
                param_sets_ids.append((name, i))
    return param_sets, param_sets_ids
