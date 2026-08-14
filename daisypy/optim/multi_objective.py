from collections.abc import Sequence
from .objective_evaluation import ObjectiveEvaluation
from .util import flatten

class MultiObjective(Sequence):
    '''Objective that computes several objectives.'''

    def __init__(self, name, objective_fns):
        '''
        Parameters
        ----------
        name : str
          Name of objective

        objective_fns : list of Callable[[str], float]
          List of objective functions. Each objective function is passed the
          path to a daisy output directory and is expected to return a named scalar objective,
          optionally via an ``evaluate`` method that also exposes predictions.
        '''
        self.name = name
        self.objective_fns = objective_fns

    def __call__(self, daisy_output_directory):
        """Compute only the scalar objective map."""
        return self.evaluate(daisy_output_directory).objectives

    def evaluate(self, daisy_output_directory):
        '''Compute the objectives and collect extracted predictions.

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory that is used when calling the objective functions

        Returns
        -------
        ObjectiveEvaluation
          Structured result containing all scalar objectives and any predictions exposed by the
          child objective functions.
        '''
        objectives = {}
        predictions = {}
        for objective_fn in self.objective_fns:
            if hasattr(objective_fn, 'evaluate'):
                evaluation = objective_fn.evaluate(daisy_output_directory)
            else:
                evaluation = ObjectiveEvaluation(objectives=objective_fn(daisy_output_directory))
            objectives.update(evaluation.objectives)
            predictions.update(evaluation.predictions)
        return ObjectiveEvaluation(objectives=objectives, predictions=predictions)

    def __getitem__(self, index):
        return self.objective_fns[index]

    def __len__(self):
        return len(self.objective_fns)

    @property
    def variable_name(self):
        '''Names of variables used in aggregated objectives

        Returns
        -------
        list of str
        '''
        return flatten(self.objective_fns, lambda x : x.variable_name)

    @property
    def target(self):
        '''Targets used on aggregated objectives

        Returns
        -------
        list of pandas.DataFrame
        '''
        return flatten(self.objective_fns, lambda x : x.target)

    @property
    def log_name(self):
        '''Names of Daisy log files used in aggregated objectives

        Returns
        -------
        list of str
        '''
        return flatten(self.objective_fns, lambda x : x.log_name)
