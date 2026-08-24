import pandas as pd
from daisypy.optim.loss_wrapper import LossWrapper
from daisypy.optim.objective import Objective
from daisypy.optim.util import check_target

class ScalarObjective(Objective):
    # pylint: disable=too-few-public-methods
    """Scalar objective"""

    def __init__(self, name, target, target_col, outcome_name, loss_fn):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Parameters
        ----------
        name : str
          Name of objective

        target : Pathlike OR pandas.DataFrame
          Either a path to csv file with the target or a DataFrame with the target
          The target DataFrame must have a "time" column with unique timestamps and atleast one
          other column.

        target_col : str
          The target column to use when computing the loss

        outcome_name : str
          Name of the outcome to use when computing the loss

        loss_fn : Callable [numpy.ndarray, numpy.ndarray] -> float
          Compute a scalar valued loss

        """
        self._name = name
        self._outcome_name = outcome_name
        if not isinstance(target, pd.DataFrame):
            target = pd.read_csv(target, sep=None, engine='python')
        check_target(target, target_col)
        self._target = target[["time", target_col]].rename(columns={target_col : 'value'})
        self._target["time"] = pd.to_datetime(self._target["time"])
        self._loss_fn = LossWrapper(loss_fn) # Wrap it so target and actual are processed correctly

    def __call__(self, outcomes):
        """Compute the objective value

        Parameters
        ----------
        outcomes : { str : pandas.DataFrame }
          A dict of named DataFrames. MUST contain the key `self.outcome_name` and the corresponding
          DataFrame MUST have columns "time" and "value"

        Returns
        -------
        { str : float }
          A dict of length 1 with the key `self.name` mapping to the objective value
        """
        return {
            self._name : self._loss_fn(actual=outcomes[self._outcome_name], target=self._target)
        }

    @property
    def target(self):
        """The target as a pandas.DataFrame with keys 'time' and 'value'"""
        return self._target

    @property
    def name(self):
        """Name of the objective"""
        return self._name

    @property
    def outcome_name(self):
        """Name of the outcome"""
        return self._outcome_name
