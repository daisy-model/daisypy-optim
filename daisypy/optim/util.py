'''Utility functions'''
from collections.abc import Sequence

__all__ = [
    'flatten'
    ]

def flatten(xs, f=None):
    '''Recursively flatten a sequence.

    Parameters
    ----------
    xs : Sequence
      The Sequence to flatten.

    f : Callable
      If not None apply f to all elements of the flattened list

    Returns
    -------
    Flattened representation of xs as a list

    Example
    -------
    >>> flatten([1, [2, 3, [4],], [[[5],6]]])
    [1, 2, 3, 4, 5, 6]

    >>> from collections.abc import Sequence
    ... class A:
    ...     def __init__(self, name):
    ...         self.name = name
    ...
    ... class B(Sequence):
    ...     def __init__(self, as_):
    ...         self.as_ = as_
    ...
    ...     @property
    ...     def name(self):
    ...         return flatten(self.as_, lambda a: a.name)
    ...
    ...     def __getitem__(self, index):
    ...         return self.As[index]
    ...
    ...     def __len__(self):
    ...         return len(self.As)
    ...
    >>> a1, a2, a3 = A('1'), A('2'), A('3')
    >>> print(a1.name, a2.name, a3.name)
    1 2 3

    >>> b1 = B([a1, a2]); b2 = B([a2, a3]); b3 = B([b1, b2])
    >>> print(b1.name, b2.name, b3.name)
    ['1', '2'] ['2', '3'] ['1', '2', '2', '3']
    '''
    if f is None:
        return _flatten_direct(xs)
    return _flatten_func(xs, f)

def _flatten_direct(xs):
    ys = []
    for x in xs:
        if isinstance(x, Sequence) and not isinstance(x, str):
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
