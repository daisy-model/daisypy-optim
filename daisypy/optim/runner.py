import subprocess

class DaisyRunner:
    """Class that knows how to run run daisy"""

    def __init__(self, daisy_bin, daisy_home=None):
        """
        Parameters
        ----------
        daisy_bin : str
          Path to daisy binary

        daisy_home : str
          Path to daisy home directory containing lib/ and sample/
          If not None set DAISYHOME to this str when calling daisy. Otherwise dont set DAISYHOME
        """
        self.daisy_bin = daisy_bin
        self.env = {} if daisy_home is None else { "DAISYHOME" : daisy_home }

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
        subprocess.CompletedProcess
        """
        args = [
            self.daisy_bin,
            "-q",
            "-d", output_directory,
            dai_file
        ]
        return subprocess.run(args, env=self.env, check=False)


    def serialize(self):
        """Serialize this DaisyRunner object

        Returns
        -------
        dict of (parameter name, parameter value) pairs
        """
        return {
            'daisy_bin' : self.daisy_bin,
            'daisy_home' : self.env.get('DAISYHOME', None)
        }

    @staticmethod
    def unzerialize(dict_repr):
        """Unserialize a DaisyRunner

        Parameters
        ----------
        dict_repr : dict of (str, str)
          Must contain
            daisy_bin : path to daisy binary
            daisy_home : path to daisy home
        """
        return DaisyRunner(dict_repr['daisy_bin'], dict_repr['daisy_home'])
