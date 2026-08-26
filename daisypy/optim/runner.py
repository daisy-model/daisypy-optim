import os
import subprocess
from pathlib import Path

class DaisyRunner:
    """Class that knows how to run daisy"""

    def __init__(self, daisy_bin, daisy_home=None):
        """
        Parameters
        ----------
        daisy_bin : str
          Path to daisy binary

        daisy_home : str
          Path to daisy home directory containing lib/ and sample/
          If not None set DAISYHOME environment variable to daisy_home. Otherwise dont set DAISYHOME
        """
        self.daisy_bin = daisy_bin
        if daisy_home is not None:
            os.environ.update('DAISYHOME', daisy_home)

    def __call__(self, dai_file, output_directory=None):
        """Run daisy

        Parameters
        ----------
        dai_file : str
          Path to dai file to run

        output_directory : str or None
          Path to output directory, if None use the directory of the dai file as output directory

        Returns
        -------
        subprocess.CompletedProcess
        """
        if output_directory is None:
            output_directory = Path(dai_file).parent
        args = [
            self.daisy_bin,
            "-q",
            "-d", output_directory,
            dai_file
        ]
        return subprocess.run(args, check=False)
