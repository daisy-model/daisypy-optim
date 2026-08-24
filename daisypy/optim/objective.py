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

    @property
    @abstractmethod
    def name(self):
        """The name of the objective
        """

    @property
    @abstractmethod
    def outcome_name(self):
        """The name of the outcome
        """

    @property
    @abstractmethod
    def target(self):
        """The target used in the objective
        """
