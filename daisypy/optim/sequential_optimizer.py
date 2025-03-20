import random
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
from .parameter import CategoricalParameter

class DaisySequentialOptimizer:
    def __init__(self, problem, logger, options={}, number_of_processes=None):
        """Daisy optimizer using a sequential approach

        The methods starts from the initial parameters. Then it changes each parameter in turn.
        The single parameter leading to best performance is then fixed and the process repeated
        untill all parameters are fixed.
        
        
        Parameters
        ----------
        problem : DaisyProblem

        options : dict
        """
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
                # If num_samples == 2, then only the initial and lower end of the valid range is used
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
        # TODO: We run the same simulation several times. The one with all "current" parameters
        fixed = set()
        floating = {}
        current = {}
        order = []
        num_params = []
        for param in self.parameters:
            floating[param.name] = param.values
            num_params.append(len(param.values))
            current[param.name] = param.values[0]
            order.append(param.name)

        num_params = sorted(num_params)[::-1]
        max_evals = 0
        for start in range(len(num_params)):
            for n in num_params[start:]:
                max_evals += n-1
        print(f'Using at most {max_evals} function evaluations')
        
            
        # Get the current loss
        param_set = [current[name] for name in order]
        print('Evaluating initial parameters')
        current_fval = self.problem(param_set)
        iteration = 0
        print('Optimizing')
        with ProcessPoolExecutor(self.number_of_processes) as executor:
            while len(floating) > 0:
                iteration += 1
                #print(current)
                # We want to generate parameter sets where we keep all parameters, exept one, fixed
                param_sets_ids = []
                param_sets = []
                for name, values in floating.items():
                    for i, value in enumerate(values):
                        if value == current[name]:
                            # We have already computed this
                            continue
                        param_set = []
                        for param_name in order:
                            if param_name == name:
                                param_set.append(float(value))
                            else:
                                param_set.append(float(current[param_name]))
                        param_sets.append(param_set)
                        param_sets_ids.append((name, i))
                self.logger.log_scalar('Number of parameter sets', len(param_sets), iteration)
                best = np.inf
                best_idx = None
                num_failures = 0
                for i, fval in enumerate(executor.map(self.problem, param_sets)):
                    if np.isnan(fval):
                        num_failures += 1
                    elif fval < best:
                        best = fval
                        best_idx = i                    
                if best_idx is None:
                    raise RuntimeError('All simulations failed')

                self.logger.log_scalar('Failed runs', num_failures, iteration)
                self.logger.log_scalar('Best', best, iteration)
                if best > current_fval:
                    print('No improvement in objective. Choosing random parameter to fix')
                    name = random.choice(list(floating.keys()))
                    floating.pop(name)
                else:
                    current_fval = best
                    name, idx = param_sets_ids[best_idx]
                    value = floating.pop(name)[idx]
                    current[name] = value
                fixed.add(name)
                print(f'Fixing {name} to {value}')

        result = {}
        for k,v in current.items():
            result[k] = { 'best': v }
        return result

        

    #             iteration += 1
    #             # Try a couple of times if we dont get at least one non nan value
    #             for i in range(max_attempts_to_get_feasible):
    #                 X = self.optimizer.ask()
    #                 fvals = np.array(eval_all(X))
    #                 if np.any(np.isfinite(fvals)):
    #                     break
    #                 else:
    #                     print(f'All are infeasible at attempt {i}')
    #                     print(fvals)
    #             self.logger.log_samples('Sample standardized parameters',
    #                                     self.problem.parameters,
    #                                     X,
    #                                     iteration)
    #             # Not sure this works
    #             self.logger.log_samples('Sample actual parameters',
    #                                     self.problem.parameters,
    #                                     self.objective.transform(X),
    #                                     iteration,
    #                                     False)
    #             failed = np.isnan(fvals)
    #             self.logger.log_scalar('Median loss', np.median(fvals[~failed]), iteration)
    #             self.logger.log_scalar('Failed runs', failed.sum(), iteration)
    #             if np.any(failed):
    #                 # cma sets nans to the median.
    #                 # We want them to have a bigger negative influence
    #                 fvals[failed] = 2*np.max(fvals[~failed])
    #             self.optimizer.tell(X, fvals)

    #             # Log parameter distributions in the standardized space
    #             means = self.optimizer.result[5]
    #             stds = self.optimizer.result[6]
    #             self.logger.log_parameter_distributions("Parameter distribution standardized",
    #                                                     self.problem.parameters,
    #                                                     means,
    #                                                     stds,
    #                                                     iteration)
    #             # Log parameter distributions in the standardized space
    #             means = self.objective.transform(means)
    #             stds = np.array(self.objective.multiplier) * stds
    #             self.logger.log_parameter_distributions("Parameter distribution actual",
    #                                                     self.problem.parameters,
    #                                                     means,
    #                                                     stds,
    #                                                     iteration,
    #                                                     single_figure=False)
                
                
    #     status = self.optimizer.result[7]
    #     print('Termination conditions')
    #     for k, v in status.items():
    #         print(k, v)
    #     best = self.objective.transform(self.optimizer.result[0])
    #     means, stds = self.optimizer.result[5], self.optimizer.result[6]
    #     transformed = self.objective.transform(means)
    #     result = {
    #         p.name : {
    #             'best' : best[i],
    #             'mean_transformed' : transformed[i],
    #             'initial_value' : p.initial_value,
    #             'valid_range' : p.valid_range,
    #             'mean' : means[i],
    #             'std' : stds[i],
    #             'transform' : (self.objective.zero[i], self.objective.multiplier[i])
    #         } for i, p in enumerate(self.problem.parameters)
    #     }
    #     return result
    # #return self.objective.transform(means), self.objective.transform(stds)
        
    # def checkpoint(self, path):
    #     # TODO: Save the state to disk to we can resume
    #     # We need to store the original problem along with the current means and stds
    #     raise NotImplementedError("Checkpointing is not yet implemted")

    # @staticmethod
    # def from_checkpoint(path):
    #     # TODO: Read the state from disk
    #     raise NotImplementedError("Resuming from checkpoint is not yet implemented")
# if __name__ == '__main__':
#     from parameter import *
#     class Problem:
#         def __init__(self, parameters):
#             self.parameters = parameters

#     parameters = [
#         CategoricalParameter('a', [1,2,3], 0),
#         ContinuousParameter('b', 0.5, (-1, 2)),
#         CategoricalParameter('c', [10,20], 1),
#     ]
#     problem = Problem(parameters)
#     optimizer = DaisySequentialOptimizer(problem, None)
#     optimizer.optimize()
