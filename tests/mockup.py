# pylint: disable=too-few-public-methods
from subprocess import CompletedProcess
from daisypy.optim.file_generator import FileGenerator

class MockFileGenerator(FileGenerator):
    '''Mock file generator that always generates the paths it was constructed with'''
    def __init__(self, paths):
        self.paths = paths

    def __call__(self, output_directory, params, tagged=True):
        return self.paths


class MockRunner:
    '''Mock runner always returning a CompletedProcess with a specified returncode'''
    def __init__(self, args=None, returncode=0):
        self.args = args if args is not None else []
        self.returncode = returncode

    def __call__(self, dai_file, output_directory):
        return CompletedProcess(self.args, self.returncode)

class MockObjective:
    '''Mock objective always returning a specific value'''
    def __init__(self, name='mock', value=0):
        self.name = name
        self.value = value

    def __call__(self, daisy_output_directory):
        return { self.name : self.value }


class MockProblem:
    '''Problem case for test purposes. Will evaluate an objective by forwarding parameters'''
    def __init__(self, parameters, objective_fn):
        self.parameters = parameters
        self.objective_fn = objective_fn

    def __call__(self, parameter_values):
        named_parameters = { p.name : value for p, value in zip(self.parameters, parameter_values) }
        return { 'mock' : self.objective_fn(**named_parameters) }

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
