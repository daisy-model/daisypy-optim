# pylint: disable=too-few-public-methods
class ContinuousParameter:
    """Wrapper for continuous Daisy parameters"""
    def __init__(self, name, initial_value, valid_range):
        """
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
    """Wrapper for categorical parameters"""
    def __init__(self, name, values, initial_value_idx=0):
        """
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
        
