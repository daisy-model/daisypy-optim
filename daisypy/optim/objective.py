# pylint: disable=too-few-public-methods
from abc import ABC, abstractmethod

class Objective(ABC):
    """Objective function interface """
    @abstractmethod
    def __call__(self, outcomes):
        """Evaluate the objective

        Parameters
        ----------
        outcomes : { str : pandas.DataFrame }
          A dict of named DataFrames. Each DataFrame has a "time" column with unique timestamps and
          one or more value columns

        Returns
        -------
        { str : float }
          A dict with named objective values
        """
