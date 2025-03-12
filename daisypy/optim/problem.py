import tempfile
import numpy as np

from daisypy.optim.objective import DaisyObjective
from daisypy.optim.runner import DaisyRunner

class DaisyOptimizationProblem:
    def __init__(self, daisy_bin, daisy_home, dai_template, log_name, parameter_name, target_value, loss_fn, data_dir=None):
        self.dai_file_generator = DaiFileGenerator(dai_template)
        self.runner = DaisyRunner(daisy_bin, daisy_home)
        self.objective = DaisyObjective(log_name, parameter_name, target_value, loss_fn)
        self.data_dir = data_dir

    def __call__(self, params):
        """
        Parameters
        ----------
        params : sequence
          Parameter values
        
        """
        with tempfile.TemporaryDirectory(dir=self.data_dir) as output_directory:
            dai_file = self.dai_file_generator(output_directory, params)
            sim_result = self.runner(dai_file, output_directory)
            if sim_result.returncode != 0:
                return np.inf
            return objective(output_directory)
        
    
