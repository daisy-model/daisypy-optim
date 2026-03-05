from collections.abc import Sequence
from .util import flatten
from .multi_objective import MultiObjective

class AggregateObjective(Sequence):
    '''Objective that computes several objectives and optionally aggregates them.'''

    def __init__(self, name, objective_fns, aggregate_fn=None):
        '''
        Parameters
        ----------
        name : str
          Name of objective

        objective_fns : dict of [str, Callable[[str], float]
          Mapping from objective names to objective functions. The objective function is passed the
          path to a daisy output directory and is expected to return a scalar

        aggregate_fn : Callable[[dict of [str, float]], float]
          Function that aggregates the computed objectives. It should map a dict of named objective
          values to a single scalar.
        '''
        self.name = name
        self.multi_objective = MultiObjective('to-be-aggregated', objective_fns)
        self.aggregate_fn = aggregate_fn

    def __call__(self, daisy_output_directory):
        '''Compute the objective

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory that is used when calling the objective functions

        Returns
        -------
        objective_map : dict of [str, float]
          Map from the objective name to the aggregated objective value
        '''
        return { self.name : self.aggregate_fn(self.multi_objective(daisy_output_directory)) }

    def __getitem__(self, index):
        # We could consider flattening objective_fns, but not sure that there is a user case
        # If we do, we also need to update __len__
        return self.multi_objective[index]

    def __len__(self):
        return len(self.multi_objective)

    @property
    def objective_fns(self):
        '''The objective functions being aggregated'''
        return self.multi_objective.objective_fns

    @property
    def variable_name(self):
        '''Names of variables used in aggregated objectives

        Returns
        -------
        list of str
        '''
        return flatten(self.multi_objective, lambda x : x.variable_name)

    @property
    def target(self):
        '''Targets used on aggregated objectives

        Returns
        -------
        list of pandas.DataFrame
        '''
        return flatten(self.multi_objective, lambda x : x.target)

    @property
    def log_name(self):
        '''Names of Daisy log files used in aggregated objectives

        Returns
        -------
        list of str
        '''
        return flatten(self.multi_objective, lambda x : x.log_name)
