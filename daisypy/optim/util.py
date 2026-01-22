'''Utility functions'''
from collections.abc import Sequence

__all__ = [
    'flatten'
    ]

def flatten(xs, f=None):
    if f is None:
        return _flatten_direct(xs)
    return _flatten_func(xs, f)
        
def _flatten_direct(xs):
    ys = []
    for x in xs:
        if isinstance(x, Sequence) and not isinstance(x, str):
            if len(x) == 1:
                ys.append(x[0])
            else:
                for y in _flatten_direct(x):
                    ys.append(y)
        else:
            ys.append(x)
    return ys

def _flatten_func(xs, f):
    ys = []
    for x in xs:
        if isinstance(x, Sequence) and not isinstance(x, str):
            for y in _flatten_func(x, f):
                ys.append(y)
        else:
            ys.append(f(x))
    return ys



class A:
    def __init__(self, name):
        self.name = name

class B(Sequence):
    def __init__(self, As):
        self.As = As

    @property
    def name(self):
        return flatten(self.As, lambda a: a.name)

    def __getitem__(self, index):
        return self.As[index]

    def __len__(self):
        return len(self.As)
    
if __name__ == '__main__':        
    a1 = A('1')
    a2 = A('2')
    a3 = A('3')
    b1 = B([a1,a2])
    b2 = B([a2,a3])
    b3 = B([b1, b2])

    print(a1.name)
    print(a2.name)
    print(a3.name)
    print(b1.name)
    print(b2.name)
    print(b3.name)
        
