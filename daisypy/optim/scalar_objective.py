import os
import pandas as pd
from daisypy.io.dlf import read_dlf
from .loss_wrapper import LossWrapper

class ScalarObjective:
    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments
    """Scalar objective that extracts data from a daisy output file and computes a loss"""

    def __init__(self, name, log_name, variable_name, target, loss_fn):
        """
        Parameters
        ----------
        name : str
          Name of objective

        log_name : str
          Name of Daisy log file where `variable_name` is found

        variable_name : str
          Name of variable to optimize for

        target : pandas.DataFrame OR str
          If str it is opened with pandas.read_csv.
          Must contain columns "time" and `variable_name`

        loss_fn : callable : (actual, target) -> loss
          The loss function to use
        """
        self.name = name
        self.log_name = log_name
        self.variable_name = variable_name
        if not isinstance(target, pd.DataFrame):
            target = pd.read_csv(target)
        self.target = target[["time", variable_name]].rename(columns={variable_name : 'value'})
        self.target["time"] = pd.to_datetime(self.target["time"])
        self.loss_fn = LossWrapper(loss_fn) # Wrap it so target and actual are processed correctly

    def __call__(self, daisy_output_directory):
        """Compute the objective

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory where there MUST be a file matching `self.log_name`

        Returns
        -------
        objective_map : dict of [str, float]
          Map from the objective name to the objetive value
        """
        dlf = read_dlf(os.path.join(daisy_output_directory, self.log_name))
        actual_value = dlf.body[self.variable_name]
        time = pd.to_datetime(
            dlf.body[['year', 'month', 'mday', 'hour']].rename(columns={'mday' : 'day'})
        )
        actual = pd.DataFrame({'time' : time, 'value' : actual_value})
        return { self.name : self.loss_fn(actual, self.target) }
