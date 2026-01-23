import multiprocessing
import warnings
import numpy as np
import cma
from cma.fitness_transformations import ScaleCoordinates
from cma.optimization_tools import EvalParallel2

class DaisyCMAOptimizer:
    def __init__(self, problem, logger, cma_options=None, number_of_processes=None):
        """Daisy optimizer using the CMA-ES method from https://github.com/CMA-ES/pycma

        There are many options for cma. The most important for new users is `maxfevals`, which
        limits the number of function evaluations. By default this is set to 10, which is
        unrealisticly low. This can be overridden by setting it to np.inf to get unlimited
        evaluations or to to a finite number, e.g
          cma_options = { "maxfevals" : np.inf }
          cma_options = { "maxfevals" : 2000 }

        A simple estimate of the time it will take to compute N evaluations can be found in this way.
          time_to_run_once = <time to run one simulation with Daisy>
          total_run_time = time_to_run_once * maxfevals / number_of_compute_cores

        Parameters
        ----------
        problem : DaisyProblem

        cma_options : dict
          Options to pass on to cma. See cma.CMAOptions for details
        """
        self.problem = problem
        self.logger = logger
        if number_of_processes is None:
            self.number_of_processes = multiprocessing.cpu_count()
        else:
            self.number_of_processes = number_of_processes
        lower = []
        upper = []
        x0 = []
        for param in problem.parameters:
            lower.append(param.valid_range[0])
            upper.append(param.valid_range[1])
            x0.append(param.initial_value)
        self.objective = ScaleCoordinates(problem, lower=lower, upper=upper, from_lower_upper=(-1,1))

        # Map the initial values to optimization domain
        x0 = self.objective.inverse(x0)

        # Setup options
        if cma_options is None:
            cma_options = {}

        if "maxfevals" not in cma_options:
            warnings.warn("Max function evaluations not set, using 10")
            cma_options["maxfevals"] = 10

        if 'bounds'  in cma_options:
            warnings.warn("'bounds' set in cma_options will be ignored and set to match problem parameters")
        cma_options['bounds'] = [-1, 1]
        self.optimizer = cma.CMAEvolutionStrategy(x0, 1/3, cma_options)

    def optimize(self):
        max_attempts_to_get_feasible = 3
        # TODO: Implement logging + checkpointing every n'th step
        total_f_evals = 0
        with EvalParallel2(self.objective, self.number_of_processes) as eval_all:
            step = 0
            while not self.optimizer.stop():
                step += 1
                # Try a couple of times if we dont get at least one non nan value
                for i in range(max_attempts_to_get_feasible):
                    X = self.optimizer.ask()
                    fvals = np.array(eval_all(X))
                    total_f_evals += len(fvals)
                    if np.any(np.isfinite(fvals)):
                        break
                    else:
                        self.logger.warning(step=step,msg=f'All are infeasible at attempt {i}', fvals=fvals)
                for x, fval in zip(X, fvals):
                    params = {
                        p.name : value  for p, value in
                        zip(self.problem.parameters, self.objective.transform(x))
                    }
                    self.logger.result(step=step, objective_value=fval, **params)

                failed = np.isnan(fvals)
                if np.all(failed):
                    self.logger.error('All attempts failed. Aborting')
                    break

                self.logger.info(step=step, total_function_evaluations=total_f_evals)
                self.logger.info(step=step, median_objective=np.median(fvals[~failed]))
                num_failures = failed.sum()
                if num_failures > 0:
                    self.logger.warning(step=step, n_failed_runs=num_failures)
                    # cma sets nans to the median.
                    # We want them to have a bigger negative influence
                    # TODO: This assumes that are we minimizing ...
                    fvals[failed] = 2*np.max(fvals[~failed])
                self.optimizer.tell(X, fvals)

                # Log parameter distributions in the standardized space
                means = self.optimizer.result[5]
                stds = self.optimizer.result[6]
                p_mean = {
                    f'{p.name}.mean' : mean for p, mean in zip(self.problem.parameters, means)
                }
                p_std = {
                    f'{p.name}.std' : std for p, std in zip(self.problem.parameters, stds)
                }
                self.logger.parameters(tag="standardized",
                                       step=step,
                                       **p_mean,
                                       **p_std)

                # Log parameter distributions in the standardized space
                means = self.objective.transform(means)
                stds = np.array(self.objective.multiplier) * stds
                p_mean = {
                    f'{p.name}.mean' : mean for p, mean in zip(self.problem.parameters, means)
                }
                p_std = {
                    f'{p.name}.std' : std for p, std in zip(self.problem.parameters, stds)
                }
                self.logger.parameters(tag="actual",
                                       step=step,
                                       **p_mean,
                                       **p_std)

        status = self.optimizer.result[7]
        self.logger.info('Termination conditions')
        for k, v in status.items():
            self.logger.info(f'{k} = {v}')
        best = self.objective.transform(self.optimizer.result[0])
        means, stds = self.optimizer.result[5], self.optimizer.result[6]
        transformed = self.objective.transform(means)
        result = {
            p.name : {
                'best' : best[i],
                'mean_transformed' : transformed[i],
                'initial_value' : p.initial_value,
                'valid_range' : p.valid_range,
                'mean' : means[i],
                'std' : stds[i],
                'transform' : (self.objective.zero[i], self.objective.multiplier[i])
            } for i, p in enumerate(self.problem.parameters)
        }
        return result

    def checkpoint(self, path):
        # TODO: Save the state to disk to we can resume
        # We need to store the original problem along with the current means and stds
        raise NotImplementedError("Checkpointing is not yet implemented")

    @staticmethod
    def from_checkpoint(path):
        # TODO: Read the state from disk
        raise NotImplementedError("Resuming from checkpoint is not yet implemented")
