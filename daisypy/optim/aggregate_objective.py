class AggregateObjective:
    '''Objective that computes several objectives and then aggregates them.'''

    def __init__(self, objective_fns, aggregate_fn):
        '''
        Parameters
        ----------
        objective_fns : list of Callable[[str], float]
          List of objective functions to compute and aggregate

        aggregate_fn : Callable[[float], float]
          Function that aggregates the computed objectives
        '''
        self.objective_fns = objective_fns
        self.aggregate_fn = aggregate_fn

    def __call__(self, daisy_output_directory):
        '''Compute the objective

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory that is used when calling the objective functions

        Returns
        -------
        objective : The aggregated objective value
        '''
        return self.aggregate_fn([f(daisy_output_directory) for f in self.objective_fns])
