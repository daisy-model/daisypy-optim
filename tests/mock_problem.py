# pylint: disable=too-few-public-methods
class MockProblem:
    '''Problem case for test purposes. Will evaluate an objective by forwarding parameters'''
    def __init__(self, parameters, objective_fn):
        self.parameters = parameters
        self.objective_fn = objective_fn

    def __call__(self, parameter_values):
        named_parameters = { p.name : value for p, value in zip(self.parameters, parameter_values) }
        return self.objective_fn(**named_parameters)
