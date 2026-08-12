# pylint: disable=R0801
import multiprocessing
import warnings
import numpy as np
import cma
from cma.fitness_transformations import ScaleCoordinates
from cma.optimization_tools import EvalParallel2
from .outcome_logging import log_outcomes
from .problem import EvaluationProblemWrapper, ScalarProblemWrapper

class DaisyCMAOptimizer:
    """Daisy optimizer using the CMA-ES method from https://github.com/CMA-ES/pycma

     There are many options for cma. The most important for new users is `maxfevals`, which
     limits the number of function evaluations. By default this is set to 10, which is
     unrealisticly low. This can be overridden by setting it to np.inf to get unlimited
     evaluations or to to a finite number, e.g
       cma_options = { "maxfevals" : np.inf }
       cma_options = { "maxfevals" : 2000 }

     A simple estimate of the time it will take to compute N evaluations can be found in this way
       time_to_run_once = <time to run one simulation with Daisy>
       total_run_time = time_to_run_once * maxfevals / number_of_compute_cores
    """
    def __init__(self, problem, logger, cma_options=None, number_of_processes=None):
        """
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
        self.objective = ScaleCoordinates(
            EvaluationProblemWrapper(problem), lower=lower, upper=upper, from_lower_upper=(-1,1)
        )

        # Map the initial values to optimization domain
        x0 = self.objective.inverse(x0)

        # Setup options
        if cma_options is None:
            cma_options = {}

        if "maxfevals" not in cma_options:
            warnings.warn("Max function evaluations not set, using 10")
            cma_options["maxfevals"] = 10

        if 'bounds'  in cma_options:
            warnings.warn(
                "'bounds' set in cma_options will be ignored and set to match problem parameters"
            )
        cma_options['bounds'] = [-1, 1]
        self.optimizer = cma.CMAEvolutionStrategy(x0, 1/3, cma_options)
        self.termination_criteria = (
            'ftarget',
            'maxfevals',
            'maxiter',
            'tolfacupx',
            'tolx',
            'tolfun',
            'tolfunrel',
            'tolfunhist',
            'tolstagnation',
            'tolxstagnation',
            'tolupsigma',
            'timeout',
            'tolconditioncov',
            'tolflatfitness',
        )

    def optimize(self):
        '''Run the optimizer'''
        # pylint: disable=too-many-locals
        max_attempts_to_get_feasible = 3
        # TODO: Implement logging + checkpointing every n'th step
        total_f_evals = 0
        self._log_termination_criteria()
        with EvalParallel2(self.objective, self.number_of_processes) as eval_all:
            step = 0
            while not self.optimizer.stop():
                step += 1
                # Try a couple of times if we dont get at least one non nan value
                for i in range(max_attempts_to_get_feasible):
                    xs = self.optimizer.ask()
                    evaluations = list(eval_all(xs))
                    fvals = np.array([
                        ScalarProblemWrapper.objective_value_from_map(evaluation.objectives)
                        for evaluation in evaluations
                    ])
                    total_f_evals += len(fvals)
                    if np.any(np.isfinite(fvals)):
                        break
                    self.logger.warning(
                        step=step,msg=f'All are infeasible at attempt {i}', fvals=fvals
                    )
                for sample_index, (x, fval, evaluation) in enumerate(zip(xs, fvals, evaluations)):
                    raw_params = {
                        f'param_{p.name}' : value  for p, value in
                        zip(self.problem.parameters, self.objective.transform(x))
                    }
                    standardized_params = {
                        f'param_{p.name}' : value  for p, value in
                        zip(self.problem.parameters, x)
                    }
                    objective_value = { f'metric_{self.problem.objective_fn.name}' : fval }
                    evaluation_id = f'{step}:{sample_index}'
                    self.logger.result(
                        evaluation_id=evaluation_id,
                        step=step,
                        tag="raw",
                        **objective_value,
                        **raw_params
                    )
                    self.logger.result(
                        evaluation_id=evaluation_id,
                        step=step,
                        tag="standardized",
                        **objective_value,
                        **standardized_params
                    )
                    log_outcomes(
                        self.logger,
                        evaluation,
                        evaluation_id=evaluation_id,
                        step=step,
                    )

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
                self.optimizer.tell(xs, fvals)

                # Log parameter distributions in the standardized space
                means = self.optimizer.result.xfavorite
                covariance = self._sampling_covariance()
                p_mean = self._means_to_columns(means)
                p_covariance = self._covariance_to_columns(covariance)
                self.logger.parameters(
                    distribution="multivariate_normal",
                    tag="standardized",
                    step=step,
                    **p_mean,
                    **p_covariance
                )

                # Log parameter distributions in the raw space
                means = self.objective.transform(means)
                # The raw parameters are an element-wise linear scaling of the standardized
                # CMA coordinates. A covariance matrix transforms as A @ C @ A.T. Here A is
                # diagonal, so this becomes an element-wise multiplication by the outer product
                # of the scaling factors.
                covariance = np.outer(self.objective.multiplier, self.objective.multiplier)
                covariance = covariance * self._sampling_covariance()
                p_mean = self._means_to_columns(means)
                p_covariance = self._covariance_to_columns(covariance)
                self.logger.parameters(
                    distribution="multivariate_normal",
                    tag="raw",
                    step=step,
                    **p_mean,
                    **p_covariance
                )

        status = self.optimizer.result.stop
        self._log_termination_criteria(status)
        best = self.objective.transform(self.optimizer.result.xbest)
        means, stds = self.optimizer.result.xfavorite, self.optimizer.result.stds
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

    def _means_to_columns(self, means):
        return {
            f'param_{p.name}_mean' : mean for p, mean in zip(self.problem.parameters, means)
        }

    def _sampling_covariance(self):
        # pycma stores the sampling distribution as
        #
        #   x = mean + sigma * sigma_vec * y,   y ~ N(0, sm.C)
        #
        # where sm.C is the normalized covariance "shape" matrix, sigma is the global
        # step-size, and sigma_vec is an element-wise linear scaling. Therefore the full
        # covariance of the standardized sampling distribution is
        #
        #   sigma^2 * D @ sm.C @ D
        #
        # with D the diagonal matrix represented by sigma_vec. pycma exposes this diagonal
        # transform via sigma_vec.transform_covariance_matrix(...).
        covariance = self.optimizer.sm.C.copy()
        covariance = self.optimizer.sigma_vec.transform_covariance_matrix(covariance)
        return self.optimizer.sigma**2 * covariance

    def _log_termination_criteria(self, status=None):
        if status is None:
            self.logger.info('Configured termination criteria')
            for criterion in self.termination_criteria:
                self.logger.info(
                    termination_criterion=criterion,
                    threshold=self.optimizer.opts[criterion]
                )
            return

        self.logger.info('Termination criteria status')
        for criterion in self.termination_criteria:
            self.logger.info(
                termination_criterion=criterion,
                threshold=self.optimizer.opts[criterion],
                current_value=self._termination_criterion_value(criterion),
                triggered=criterion in status
            )

    def _termination_criterion_value(self, criterion):
        if criterion == 'ftarget':
            return self.optimizer.best.f
        if criterion == 'maxfevals':
            return self.optimizer.countevals - 1
        if criterion == 'maxiter':
            return self.optimizer.countiter
        if criterion == 'tolfacupx':
            coordinate_stds = self._standardized_coordinate_stds()
            reference = np.atleast_1d(self.optimizer.sigma0) * np.atleast_1d(self.optimizer.sigma_vec0)
            return np.max(coordinate_stds / reference)
        if criterion == 'tolfun':
            current_fitness_range = max(self.optimizer.fit.fit) - min(self.optimizer.fit.fit)
            historic_fitness_range = max(self.optimizer.fit.hist) - min(self.optimizer.fit.hist)
            return {
                'current_fitness_range' : current_fitness_range,
                'historic_fitness_range' : historic_fitness_range,
            }
        if criterion == 'tolfunhist':
            return max(self.optimizer.fit.hist) - min(self.optimizer.fit.hist)
        if criterion == 'tolstagnation':
            window = max((
                self.optimizer.opts['tolstagnation'] / 5. / 2,
                len(self.optimizer.fit.histbest) / 10
            ))
            if window > self.optimizer.countiter:
                return {
                    'window' : window,
                    'countiter' : self.optimizer.countiter,
                }
            window = int(window)
            return {
                'window' : window,
                'median_history_previous' : np.median(self.optimizer.fit.histmedian[:window]),
                'median_history_recent' : np.median(self.optimizer.fit.histmedian[window:2 * window]),
                'best_history_previous' : np.median(self.optimizer.fit.histbest[:window]),
                'best_history_recent' : np.median(self.optimizer.fit.histbest[window:2 * window]),
            }
        if criterion == 'tolxstagnation':
            return {
                'count' : self.optimizer._stoptolxstagnation.count,
                'count_x' : self.optimizer._stoptolxstagnation.count_x,
                'time_threshold' : self.optimizer._stoptolxstagnation.time_threshold,
            }
        if criterion == 'timeout':
            if hasattr(self.optimizer, 'timer'):
                return self.optimizer.timer.elapsed
            return None
        if criterion == 'tolconditioncov':
            return self.optimizer.D[-1]**2 / self.optimizer.D[0]**2
        if criterion == 'tolflatfitness':
            return self.optimizer.fit.flatfit_iterations
        return self.optimizer.stop(check=False, get_value=criterion)

    def _standardized_coordinate_stds(self):
        return self.optimizer.sigma * (
            self.optimizer.sigma_vec.scaling * np.sqrt(self.optimizer.dC)
        )

    def _covariance_to_columns(self, covariance):
        columns = {}
        for i, row_parameter in enumerate(self.problem.parameters):
            for j, column_parameter in enumerate(self.problem.parameters[i:], start=i):
                columns[f'param_{row_parameter.name}__param_{column_parameter.name}_cov'] = (
                    covariance[i, j]
                )
        return columns

    def checkpoint(self, path):
        '''Save the state to disk to we can resume

        Parameters
        ----------
        path : str
          Path to store checkpoint in
        '''
        # TODO: Save state to disk
        raise NotImplementedError("Checkpointing is not yet implemented")

    @staticmethod
    def from_checkpoint(path):
        '''Read state from disk to we can resume

        Parameters
        ----------
        path : str
          Path to read checkpoint from
        '''
        # TODO: Read the state from disk
        raise NotImplementedError("Resuming from checkpoint is not yet implemented")
