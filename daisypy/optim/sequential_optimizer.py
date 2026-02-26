from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .parameter import CategoricalParameter

class DaisySequentialOptimizer:
    def __init__(self, problem, logger, options=None, number_of_processes=None):
        """Daisy optimizer using a sequential approach

        The methods starts from the initial parameters. Then it changes each parameter in turn.
        The single parameter leading to best performance is then fixed and the process repeated
        untill all parameters are fixed.


        Parameters
        ----------
        problem : DaisyProblem

        logger : ...

        options : dict

        number_of_processes: int > 0 (Optional)
          The maximum number of processes to use when running Daisy. Defaults to
          os.process_cpu_count()
        """
        if options is None:
            options = {}
        self.problem = problem
        self.logger = logger
        self.number_of_processes = number_of_processes

        # Convert any continuous parameters to categorical parameters by uniform sampling
        num_samples = options.get("num_samples", 3)
        self.parameters = []
        for param in problem.parameters:
            # Standardize parameters such that they are all categorical and the initial value is the
            # first value in the values list.
            if param.type == "Continuous":
                # If num_samples == 2, then only the initial and lower end of the valid range is
                # used.
                # We can end up with the initial_value being added twice, if for example we have
                # valid_range = (0, 10)
                # initial_value = 5
                # num_samples = 12
                # It is not clear what is the best solution to this. If we sample the ranges
                # (low, initial) and (initial, high) separately then we get different spacing in
                # most cases.
                # For now we just ignore it and accept that we sometimes try one less parameter than
                # we would like.
                values = np.concatenate([
                    [param.initial_value],
                    np.linspace(param.valid_range[0], param.valid_range[1], num_samples-1)
                ])
                self.parameters.append(CategoricalParameter(param.name, values, 0))
            elif param.type == 'Categorical':
                if param.initial_value_idx != 0:
                    values = np.concatenate([
                        [param.values[param.initial_value_idx]],
                        param.values[:param.initial_value_idx],
                        param.values[param.initial_value_idx+1:]
                    ])
                    param = CategoricalParameter(param.name, values, 0)
                self.parameters.append(param)

    def optimize(self):
        '''Run optimization'''
        # pylint: disable=too-many-locals,too-many-statements
        # Recall that we are working with categorical parameters, so there is no sampling of new
        # parameters.
        step = 0
        fixed = set()   # The parameters that are already fixed
        floating = {}   # The parameters that we need to fix
        current = {}    # The parameter values that we are currently using
        order = []      # Order that parameters are passed to the problem.
        num_param_values = [] # Number of possible parameter values for each parameter
        for param in self.parameters:
            floating[param.name] = param.values
            current[param.name] = param.values[0] # Parameter values are tried in order
            order.append(param.name)
            num_param_values.append(len(param.values))

        min_evals, max_evals = _count_min_max_param_evals(num_param_values)
        self.logger.info(f'Using at least {min_evals} and at most {max_evals} function evaluations')

        # Compute the initial loss
        self.logger.info('Evaluating initial parameters')
        current_fval = self.problem([current[name] for name in order])
        if np.isnan(current_fval):
            self.logger.error('Initial parameters failed, aborting')
            raise RuntimeError('Initial parameters failed')

        self.logger.info(f'Initial objective = {current_fval}')
        total_f_evals = 1
        self.logger.info('Optimizing')
        with ProcessPoolExecutor(self.number_of_processes) as executor:
            while len(floating) > 0:
                # We fix a parameter in each step, so we will always do as many steps as there are
                # parameters.
                step += 1

                param_sets, param_sets_ids = _generate_parameter_sets(floating, current, order)
                self.logger.info(step=step, n_param_sets=len(param_sets))

                best = np.inf
                best_idx = None
                num_failures = 0
                # executor.map runs the problems in parallel and yields results in order matching
                # param_sets.
                for i, fval in enumerate(executor.map(self.problem, param_sets)):
                    self.logger.result(
                        step=step, objective_value=fval, **dict(zip(order, param_sets[i]))
                    )
                    if np.isnan(fval):
                        num_failures += 1
                    elif fval < best:
                        best = fval
                        best_idx = i # Index into param_sets
                if best_idx is None:
                    # Maybe not raise an exception if we have had at least one successful run in a
                    # previous step?
                    self.logger.error('All simulations failed. Aborting')
                    raise RuntimeError('All simulations failed')

                total_f_evals += len(param_sets)
                self.logger.info(step=step, total_function_evaluations=total_f_evals)
                if num_failures > 0:
                    self.logger.warning(step=step, n_failed_runs=num_failures)
                self.logger.info(step=step, best_objective=best)
                if best > current_fval:
                    # Nothing is better than using current values of all parameters, so we stop.
                    # We could consider setting a random parameter to a random value, or something
                    # similar.
                    self.logger.info('No improvement in objective. Stopping')
                    break

                current_fval = best
                name, idx = param_sets_ids[best_idx]
                value = floating.pop(name)[idx]
                current[name] = value
                fixed.add(name)
                self.logger.info(f'step={step},Fixing {name} to {value}')

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
    for start in range(len(num_param_values)):
        for n in num_param_values[start:]:
            min_evals += n-1
    return min_evals, max_evals

def _generate_parameter_sets(floating, current, order):
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
                    param_set.append(float(value))
                else:
                    param_set.append(float(current[param_name]))
            param_sets.append(param_set)
            param_sets_ids.append((name, i))
    return param_sets, param_sets_ids
