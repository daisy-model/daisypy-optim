from collections.abc import Sequence
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
          List of objective functions. The objective function is passed the
          path to a daisy output directory and is expected to return a scalar
        '''
        self.name = name
        self.objective_fns = objective_fns

    def __call__(self, daisy_output_directory):
        '''Compute the objectives

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory that is used when calling the objective functions

        Returns
        -------
        objective_map : dict of [str, float]
          Mapping from objective names to objective values
        '''
        return { f.name : f(daisy_output_directory) for f in self.objective_fns }

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
