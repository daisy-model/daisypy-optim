# pylint: disable=too-few-public-methods
from subprocess import CompletedProcess
import pandas as pd
from daisypy.optim.file_generator import FileGenerator
from daisypy.optim import ScalarObjective

class MockError:
    def __init__(self, returncode=1, msg="FAIL"):
        self.returncode = returncode
        self.msg = msg

class MockFileGenerator(FileGenerator):
    '''Mock file generator that always generates the paths it was constructed with'''
    def __init__(self, path):
        self.path = path

    def __call__(self, output_directory, params, tagged=True):
        return self.path

    def relative_out_path(self):
        return self.path

    def copy_and_update(self, **kwargs):
        return MockFileGenerator(kwargs.get("path", self.path))


class MockRunner:
    '''Mock runner always returning a CompletedProcess with a specified returncode'''
    def __init__(self, args=None, returncode=0):
        self.args = args if args is not None else []
        self.returncode = returncode

    def __call__(self, dai_file, output_directory):
        return CompletedProcess(self.args, self.returncode)

class MockObjective(ScalarObjective):
    '''Mock objective always returning a specific value'''
    def __init__(self, name="mock", value=0):
        self.name = name
        self.outcome_name = "MockObjective.outcome"
        self.target = pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01']),
            'value' : [42]
        })
        self.value = value

    def __call__(self, daisy_output_directory):
        return { self.name : self.value }


class MockProblem:
    '''Problem case for test purposes. Will evaluate an objective by forwarding parameters'''
    def __init__(self, parameters, objective_fn, error=None):
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.error = {} if error is None else error

    def __call__(self, parameter_values):
        '''Evaluate objective and return value and outcomes'''
        if len(self.error):
            return ( {}, {}, self.error )
        named_parameters = { p.name : value for p, value in zip(self.parameters, parameter_values) }
        objective_value = self.objective_fn(**named_parameters)
        prediction = pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01']),
            'value' : [objective_value]
        })
        return (
            { self.objective_fn.name : objective_value },
            { self.objective_fn.outcome_name : prediction },
            { }
        )

class MockDataExtractor:
    '''Mock data extractor returning data it was constructed with'''
    def __init__(self, data):
        self.data = data

    def __call__(self, daisy_output_directory):
        return self.data

class MockLoss:
    '''Mock loss that always returns a specificed value'''
    def __init__(self, value):
        self.value = value

    def __call__(self, actial, target):
        return self.value
