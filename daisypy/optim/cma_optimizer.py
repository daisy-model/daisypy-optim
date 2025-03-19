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
        # TODO: Implement logging + checkpointing every n'th iteration
        with EvalParallel2(self.objective, self.number_of_processes) as eval_all:
            iteration = 0
            while not self.optimizer.stop():
                iteration += 1
                # Try a couple of times if we dont get at least one non nan value
                for i in range(max_attempts_to_get_feasible):
                    X = self.optimizer.ask()
                    fvals = np.array(eval_all(X))
                    if np.any(np.isfinite(fvals)):
                        break
                    else:
                        print(f'All are infeasible at attempt {i}')
                        print(fvals)
                self.logger.log_samples('Sample standardized parameters',
                                        self.problem.parameters,
                                        X,
                                        iteration)
                # Not sure this works
                self.logger.log_samples('Sample actual parameters',
                                        self.problem.parameters,
                                        self.objective.transform(X),
                                        iteration,
                                        False)
                failed = np.isnan(fvals)
                self.logger.log_scalar('Median loss', np.median(fvals[~failed]), iteration)
                self.logger.log_scalar('Failed runs', failed.sum(), iteration)
                if np.any(failed):
                    # cma sets nans to the median.
                    # We want them to have a bigger negative influence
                    fvals[failed] = 2*np.max(fvals[~failed])
                self.optimizer.tell(X, fvals)

                # Log parameter distributions in the standardized space
                means = self.optimizer.result[5]
                stds = self.optimizer.result[6]
                self.logger.log_parameter_distributions("Parameter distribution standardized",
                                                        self.problem.parameters,
                                                        means,
                                                        stds,
                                                        iteration)
                # Log parameter distributions in the standardized space
                means = self.objective.transform(means)
                stds = np.array(self.objective.multiplier) * stds
                self.logger.log_parameter_distributions("Parameter distribution actual",
                                                        self.problem.parameters,
                                                        means,
                                                        stds,
                                                        iteration,
                                                        single_figure=False)
                
                
        status = self.optimizer.result[7]
        print('Termination conditions')
        for k, v in status.items():
            print(k, v)
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
    #return self.objective.transform(means), self.objective.transform(stds)
        
    def checkpoint(self, path):
        # TODO: Save the state to disk to we can resume
        # We need to store the original problem along with the current means and stds
        raise NotImplementedError("Checkpointing is not yet implemted")

    @staticmethod
    def from_checkpoint(path):
        # TODO: Read the state from disk
        raise NotImplementedError("Resuming from checkpoint is not yet implemented")
