class DaisyParameter:
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
        self.name = name
        self.initial_value = initial_value
        self.valid_range = valid_range
