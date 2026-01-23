'''Test functions from https://en.wikipedia.org/wiki/Test_functions_for_optimization'''
from daisypy.optim import ContinuousParameter

class BealeFunction:
    def __init__(self):
        self.parameters = [
            ContinuousParameter('x', 0, (-4,4)),
            ContinuousParameter('y', 0, (-4,4))
        ]
        self.amin = { 'x' : 3, 'y' : 0.5 }
        self.min = 0
        
    def __call__(self, x, y):
        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

beale_function = BealeFunction()
