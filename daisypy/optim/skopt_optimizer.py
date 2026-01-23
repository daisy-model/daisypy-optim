import multiprocessing
import warnings
import numpy as np
import skopt
from joblib import Parallel, delayed

class DaisySkoptOptimizer:
    def __init__(self, problem, logger, options={}, number_of_processes=None):
        """Daisy optimizer using the scikit-optimize library
        https://scikit-optimize.github.io/stable/

        Parameters
        ----------
        problem : DaisyProblem

        options : dict
          Options to pass on to the optimizer
        """
        self.problem = problem
        self.logger = logger
        if number_of_processes is None:
            self.number_of_processes = multiprocessing.cpu_count()
        else:
            self.number_of_processes = number_of_processes

        dimensions = []
        for param in problem.parameters:
            dimensions.append(param.valid_range)

        self.objective = problem

        # Setup options
        self.max_f_evals = options.pop("maxfevals", 10)
        self.optimizer = skopt.Optimizer(dimensions, **options)

    def optimize(self):
        max_attempts_to_get_feasible = 3
        step = 0
        total_f_evals = 0
        while total_f_evals < self.max_f_evals:
            step += 1
            # Try a couple of times if we dont get at least one non nan value
            for i in range(max_attempts_to_get_feasible):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = self.optimizer.ask(n_points=self.number_of_processes)
                fvals = np.array(
                    Parallel(prefer="threads", n_jobs=self.number_of_processes)(
                        delayed(self.objective)(x) for x in X
                    ))
                total_f_evals += self.number_of_processes
                if np.any(np.isfinite(fvals)):
                    break
                else:
                    self.logger.warning(
                        step=step,msg=f'All are infeasible at attempt {i}', fvals=fvals
                    )
            # Log the specific parameters and the corresponding objective values
            for x, fval in zip(X, fvals):
                params = {
                    p.name : value  for p, value in
                    zip(self.problem.parameters, x)
                }
                self.logger.result(step=step, objective_value=fval, **params)

            # If everything fails we abort
            failed = np.isnan(fvals)
            if np.all(failed):
                self.logger.error('All attempts failed. Aborting')
                break

            self.logger.info(step=step, total_function_evaluations=total_f_evals)
            self.logger.info(step=step, median_objective=np.median(fvals[~failed]))
            num_failures = failed.sum()
            if num_failures > 0:
                self.logger.warning(step=step, n_failed_runs=num_failures)
                # TODO: This assumes that are we minimizing ...
                fvals[failed] = 2*np.max(fvals[~failed])
            self.optimizer.tell(X, list(fvals))

        optim_result = self.optimizer.get_result()
        best = optim_result.x
        result = {
            p.name : {
                'best' : float(best[i]),
                'valid_range' : p.valid_range,
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
