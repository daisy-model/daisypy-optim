import multiprocessing
import warnings
import cma
from cma.fitness_transformations import ScaleCoordinates
from cma.optimization_tools import EvalParallel2

class DaisyCMAOptimizer:
    def __init__(self, problem, cma_options=None, number_of_processes=None):
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
        # TODO: Rewrite to an eplicit loop and implement logging + checkpointing every n'th iteration
        with EvalParallel2(self.objective, self.number_of_processes) as eval_all:
            while not self.optimizer.stop():
                X = self.optimizer.ask()
                self.optimizer.tell(X, eval_all(X))
                
        status = self.optimizer.result[7]
        print('Termination conditions')
        for k, v in status.items():
            print(k, v)
        means, stds = self.optimizer.result[5], self.optimizer.result[6]
        transformed = self.objective.transform(means)
        result = {
            p.name : {
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
        
    def checkpoint(self):
        # TODO: Save the state to disk to we can resume
        # We need to store the original problem along with the current means and stds
        raise NotImplementedError("Checkpointing is not yet implemted")

    @staticmethod
    def from_checkpoint(path):
        # TODO: Read the state from disk
        raise NotImplementedError("Resuming from checkpoint is not yet implemented")
    
