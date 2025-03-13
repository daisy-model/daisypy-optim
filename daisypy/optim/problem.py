import tempfile
import os
import numpy as np

class DaisyOptimizationProblem:
    def __init__(self, runner, dai_file_generator, objective_fn, parameters, data_dir=None):
        """
        Parameters
        ----------
        runner : DaisyRunner

        dai_file_generator : DaiFileGenerator

        objective_fn : DaisyObjective

        parameters : list of DaisyParameter

        data_dir : str
          If not None then temporary directories will be created in this directory. Otherwise, they
          will be created in a default location depending on platform.
        """
        self.runner = runner
        self.dai_file_generator = dai_file_generator
        self.objective_fn = objective_fn
        self.parameters = parameters
        self.data_dir = data_dir

    def __call__(self, parameter_values):
        """
        Parameters
        ----------
        parameter_values : sequence
          Parameter values. Lenght MUST match length of `self.parameters`

        """
        named_parameters = { p.name : value for p, value in zip(self.parameters, parameter_values) }
        with tempfile.TemporaryDirectory(dir=self.data_dir) as output_directory:
            dai_file = self.dai_file_generator(output_directory, named_parameters)
            sim_result = self.runner(dai_file, output_directory)
            if sim_result.returncode != 0:
                return np.nan
            return self.objective_fn(output_directory)
