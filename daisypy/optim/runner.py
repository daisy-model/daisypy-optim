import os
import subprocess

class DaisyRunner:
    def __init__(self, daisy_bin, daisy_home):
        """
        Parameters
        ----------
        daisy_bin : str
          Path to daisy binary

        daisy_home : str
          Path to daisy home directory containing lib/ and sample/
        """
        self.daisy_bin = daisy_bin
        self.daisy_home = daisy_home

    def __call__(self, dai_file, output_directory):
        """Run daisy

        Parameters
        ----------
        dai_file : str
          Path to dai file to run

        output_directory : str
          Path to output directory

        Returns
        -------
        sybprocess.CompletedProcess
        """
        args = [
            self.daisy_bin,
            "-q",
            "-d", output_directory,
            dai_file
        ]
        return subprocess.run(args, env={"DAISYHOME" : self.daisy_home})


    def serialize(self):
        return {
            'daisy_bin' : self.daisy_bin,
            'daisy_home' : self.daisy_home
        }

    @staticmethod
    def unzerialize(dict_repr):
        return DaisyRunner(dict_repr['daisy_bin'], dict_repr['daisy_home'])
