from daisypy.optim.objective import Objective

class MultiObjective(Objective):
    # pylint: disable=too-few-public-methods
    '''Simple wrapper for computing multiple objectives.

    Attributes
    ----------
    name : str
      Name of objective

    objectives : [daispy.optim.objective.Objective]
      List of objectives to compute
    '''

    def __init__(self, name, objectives, aggregate_fn=None):
        '''
        Parameters
        ----------
        objectives : [daispy.optim.objective.Objective]
          List of objectives. Each objective is passed the outcomes and is expected to return named
          scalar objective value(s). Names are assumed unique.

        aggregate_fn : Callable[[dict of [str, float]], float] or None
          Optional function that aggregates the computed objectives. It should map a dict of named
          objective values to a single scalar.
        '''
        self.name = name
        self.objectives = objectives
        self._aggregate_fn = aggregate_fn

    def __call__(self, outcomes):
        """Compute all objectives and aggregate if an aggregation function was provided

        Parameters
        ----------
        outcomes : { str : pandas.DataFrame }
          A dict of named DataFrames. Each DataFrame has a "time" column with unique timestamps and
          a "value" column with values.

        Returns
        -------
        { str : float }
          A dict with named objective values

        Raises
        ------
        ValueError if objective value names are not unique
        """
        objective_values = {}
        for objective in self.objectives:
            for k, v in objective(outcomes).items():
                if k in objective_values:
                    raise ValueError(f"Objective names must be unique: '{k}'")
                objective_values[k] = v
        if self._aggregate_fn is not None:
            return { self.name : self._aggregate_fn(objective_values) }
        return objective_values
