import os
from daisypy.io.dlf import read_dlf

class DaisyObjective:
    def __init__(self, log_name, parameter_name, target_value, loss_fn):
        self.log_name = log_name
        self.parameter_name = parameter_name
        self.target_value = target_value
        self.loss_fn = loss_fn

    def __call__(self, daisy_output_directory):
        dlf = read_dlf(os.path.join(daisy_output_directory, self.log_name))
        actual_value = dlf.body[self.parameter_name].iloc[-1]
        loss = self.loss_fn(actual_value, self.target_value)
        return loss
