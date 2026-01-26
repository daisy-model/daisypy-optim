import tempfile
import os
import platform
import numpy as np

class DaisyOptimizationProblem:
    def __init__(
            self, runner, file_generator, objective_fn, parameters, data_dir=None, debug=False
    ):
        """
        Parameters
        ----------
        runner : DaisyRunner

        file_generator : FileGenerator

        objective_fn : DaisyObjective

        parameters : list of DaisyParameter

        data_dir : str
          If not None then temporary directories will be created in this directory. Otherwise, they
          will be created in a default location depending on platform.
        """
        self.runner = runner
        self.file_generator = file_generator
        self.objective_fn = objective_fn
        self.parameters = parameters
        self.data_dir = data_dir
        if data_dir is None and platform.system().lower() == 'linux':
            # There is a good chance that we are using flatpak, in which case we need to use a tmp
            # location that flatpak Daisy can read and write. Anything inside the user home is good.
            self.data_dir = os.path.expanduser('~/.tmp/daisy')
        if self.data_dir is not None:
            os.makedirs(self.data_dir, exist_ok=True)
        self.debug = debug

    def __call__(self, parameter_values):
        """
        Parameters
        ----------
        parameter_values : sequence
          Parameter values. Lenght MUST match length of `self.parameters`

        """
        named_parameters = { p.name : value for p, value in zip(self.parameters, parameter_values) }
        # If we debug then we dont want the directory to be deleted after use
        # From python 3.12 we can pass delete=False to TemporaryDirectory, but prior to that we need
        # to use mkdtemp.
        if self.debug:
            output_directory = tempfile.mkdtemp(dir=self.data_dir)
            return self._run(output_directory, named_parameters)

        with tempfile.TemporaryDirectory(dir=self.data_dir) as output_directory:
            return self._run(output_directory, named_parameters)

    def _run(self, output_directory, named_parameters):
        dai_file = self.file_generator(output_directory, named_parameters)['dai']
        sim_result = self.runner(dai_file, output_directory)
        if sim_result.returncode != 0:
            print(sim_result)
            return np.nan
        return self.objective_fn(output_directory)
