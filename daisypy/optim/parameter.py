class ContinuousParameter:
    def __init__(self, name, initial_value, valid_range):
        """Wrapper for continuous Daisy parameters

        Parameters
        ----------
        name : str
          Name of parameter

        initial_value : float
          Initial value of parameter to use for starting the optimization

        valid_range : (float, float)
          Lowest and highest valid values for the parameter
        """
        self.type = "Continuous"
        self.name = name
        self.initial_value = initial_value
        self.valid_range = valid_range


class CategoricalParameter:
    def __init__(self, name, values, initial_value_idx=0):
        """Wrapper for categorical parameters

        Parameters
        ----------
        name : str
          Name of parameter

        values : list of values
          Possible parameter values

        initial_value_idx : int
          Index of initial parameter value
        """
        self.type = "Categorical"
        self.name = name
        self.values = values
        self.initial_value_idx = initial_value_idx
        
