import os
import pandas as pd
from daisypy.io.dlf import read_dlf

class DaisyObjective:
    def __init__(self, log_name, variable_name, target, loss_fn):
        """
        Parameters
        ----------
        log_name : str
          Name of Daisy log file where `variable_name` is found

        variable_name : str
          Name of variable to optimize for

        target : pd.DataFrame
          Must contain columns "time" and `variable_name`

        loss_fn : daisypy.optim.DaisyLoss
          The loss function to use
        """
        self.log_name = log_name
        self.variable_name = variable_name
        self.target = target[["time", variable_name]].rename(columns={variable_name : 'value'})
        self.loss_fn = loss_fn

    def __call__(self, daisy_output_directory):
        """Compute the objective

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory where there MUST be a file matching `self.log_name`

        Returns
        -------
        loss : The computed loss value
        """
        dlf = read_dlf(os.path.join(daisy_output_directory, self.log_name))
        actual_value = dlf.body[self.variable_name]
        time = pd.to_datetime(
            dlf.body[['year', 'month', 'mday', 'hour']].rename(columns={'mday' : 'day'})
        )
        actual = pd.DataFrame({'time' : time, 'value' : actual_value})
        loss = self.loss_fn(actual, self.target)
        return loss
