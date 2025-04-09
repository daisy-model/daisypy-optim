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
        iteration = 0
        total_f_evals = 0
        while total_f_evals < self.max_f_evals:
            iteration += 1
            # Try a couple of times if we dont get at least one non nan value
            for i in range(max_attempts_to_get_feasible):
                X = self.optimizer.ask(n_points=self.number_of_processes)
                fvals = np.array(Parallel(n_jobs=self.number_of_processes)(
                    delayed(self.objective)(x) for x in X
                ))
                total_f_evals += self.number_of_processes
                if np.any(np.isfinite(fvals)):
                    break
                else:
                    print(f'All are infeasible at attempt {i}')
                    print(fvals)
            self.logger.log_samples('Sample actual parameters',
                                    self.problem.parameters,
                                    X,
                                    iteration,
                                    False)
            failed = np.isnan(fvals)
            self.logger.log_scalar('Total function evaluations', total_f_evals, iteration)
            self.logger.log_scalar('Median loss', np.median(fvals[~failed]), iteration)
            self.logger.log_scalar('Failed runs', failed.sum(), iteration)
            if np.any(failed):
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
