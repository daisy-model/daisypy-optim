import pandas as pd
from .loss_wrapper import LossWrapper

class ScalarObjective:
    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments
    """Scalar objective that extracts data from a daisy output directory and computes a loss"""

    def __init__(self, name, data_extractor, target, target_name, loss_fn):
        """
        Parameters
        ----------
        name : str
          Name of objective

        data_extractor : DlfDataExtractor
          Extractor mapping output directories to pandas.Series

        target : pandas.DataFrame OR str
          If str it is opened with pandas.read_csv.
          Must contain columns "time" and `target_name`

        target_name : str
          Name of column in target that contains the target values

        loss_fn : callable : (actual, target) -> loss
          The loss function to use
        """
        self.name = name
        self.data_extractor = data_extractor
        if not isinstance(target, pd.DataFrame):
            target = pd.read_csv(target)
        self.target = target[["time", target_name]].rename(columns={target_name : 'value'})
        self.target["time"] = pd.to_datetime(self.target["time"])
        self.loss_fn = LossWrapper(loss_fn) # Wrap it so target and actual are processed correctly

    def __call__(self, daisy_output_directory):
        """Compute the objective

        Parameters
        ----------
        daisy_output_directory : str
          Path to daisy ouput directory

        Returns
        -------
        objective_map : dict of [str, float]
          Map from the objective name to the objective value
        """
        actual = self.data_extractor(daisy_output_directory)
        return { self.name : self.loss_fn(actual, self.target) }
