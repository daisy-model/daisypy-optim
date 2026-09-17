import numpy as np
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

    def sample(self, num_samples):
        '''Sample the continuous parameter range. This tries to sample the range evenly while also
        ensuring the initial value and valid range endpoitns are sampled.

        Parameters
        ----------
        num_samples : int > 0
          Number of samples to generate

        Raises
        ------
        ValueError if num_samples is not positive

        Returns
        -------
        List of float, where the first element is the initial value
        '''
        if num_samples < 1:
            raise ValueError(f'`num_samples` must be positive ({num_samples})')

        if num_samples == 1:
            return [self.initial_value]

        a, b = self.valid_range
        x = self.initial_value
        before_range = x - a
        after_range = b - x
        if num_samples == 2:
            if before_range > after_range:
                return [x, a]
            return [x, b]

        before_ratio = before_range / (b - a)
        num_before_samples = max(1, round((num_samples-1) * before_ratio))
        num_after_samples = num_samples - 1 - num_before_samples
        if num_after_samples == 0:
            num_before_samples -= 1
            num_after_samples = 1
        before_samples = [float(v) for v in np.linspace(a, x, num_before_samples + 1)[:-1]]
        after_samples = [float(v) for v in np.linspace(x, b, num_after_samples + 1)[1:]]
        return [x] + before_samples + after_samples


    def as_categorical(self, num_samples):
        '''Convert to CategoricalParameter

        Parameters
        ----------
        num_samples : int > 0
          Number of samples to get from the valid range

        See also
        --------
        self.sample

        Returns
        -------
        CategoricalParameter with the same name, num_samples samples and the initial value at the
        front of the values list.
        '''
        return CategoricalParameter(self.name, self.sample(num_samples), 0)

class CategoricalParameter:
    # pylint: disable=too-few-public-methods
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

    def normal_form(self):
        '''Returns a copy of this CategoricalParameter where the initial value is the first element
        of the values list. It is assumed that the values list does not contain compound objects.

        Returns
        -------
        CategoricalParameter with initial value at the front of values
        '''
        idx = self.initial_value_idx
        if idx != 0:
            values = [self.values[idx]] + self.values[:idx] + self.values[idx+1:]
        else:
            values = self.values.copy()
        return CategoricalParameter(self.name, values, 0)
