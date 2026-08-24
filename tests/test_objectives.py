# pylint: disable=too-few-public-methods
'''Test functions from https://en.wikipedia.org/wiki/Test_functions_for_optimization'''
import pandas as pd
from daisypy.optim import ContinuousParameter

class BealeFunction:
    '''The BealeFunction'''
    def __init__(self):
        self.name = "Beale"
        self.target = pd.DataFrame({"time" : [], "value": []})
        self.outcome_name = "Beale.outcome"
        self.parameters = [
            ContinuousParameter('x', 0, (-4,4)),
            ContinuousParameter('y', 0, (-4,4))
        ]
        self.amin = { 'x' : 3, 'y' : 0.5 }
        self.min = 0

    def __call__(self, x, y):
        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

beale_function = BealeFunction()
