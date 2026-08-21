# pylint: disable=too-few-public-methods
from abc import ABC, abstractmethod

class PostProcessor(ABC):
    """Interface for post processing functions"""

    @abstractmethod
    def __call__(self, outcomes):
        """Evaluate the post processing function on the outcomes

        Parameters
        ----------
        outcomes : { str : pandas.DataFrame }
          Dict of named outcomes. Each DataFrame has columns "time" and "value". It is the
          responsibility of the implementer to ensure that any operation on dataframes makes
          sense. For exampl, that outcomes with different time stamps are not merged
          unintentionally.

        Returns
        -------
        pandas.DataFrame with columns "time" and "value"
        """
