# pylint: disable=too-few-public-methods
from abc import ABC, abstractmethod

class FileGenerator(ABC):
    """Generate a file in the given output directory using the given params"""
    @abstractmethod
    def __call__(self, output_directory, params):
        """
        Parameters
        ----------
        output_directory : str
          Directory to store the generated file in

        params : dict
          Dictionary of parameters.

        Returns
        -------
        paths : { str : str }
          Map from generator type to generated file paths
        """
