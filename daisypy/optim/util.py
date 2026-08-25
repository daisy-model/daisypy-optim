'''Utility functions'''
from collections.abc import Sequence
from string import Formatter
import shutil
import pandas as pd

__all__ = [
    'flatten',
    'copy_into',
    "merge_outcomes",
    "check_outcomes",
    "check_target",
    "get_single_scalar",
    "StrictFormatter",
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


def copy_into(src, dst_dir):
    """Copy src into dst_dir. This mimics Path.copy_into that is introduced in python 3.14"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst = dst_dir / src.name
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy(src, dst_dir)


def check_target(target, target_col):
    """Check that a target DataFrame is valid. A target is valid if it has a "time" column with
    unique values and a column named `target_col`

    Parameters
    ----------
    target : pandas.DataFrame

    target_col : str

    Raises
    ------
    ValueError if something is wrong with the target
    """
    if "time" not in target.columns:
        raise ValueError("Missing 'time' column")
    if target_col not in target.columns:
        raise ValueError(f"Missing '{target_col}' column")
    if target["time"].nunique() != len(target):
        raise ValueError("Time points are not unique")


def merge_outcomes(outcomes):
    """Merge a set of outcomes. Note that the result is not a valid outcome, it is a DataFrame with
    a column for each outcome.

    Precondition: check_outcomes(outcomes) does not throw.

    Parameters
    ----------
    outcomes : { str : pandas.DataFrame(s) }
      One or more named outcomes to merge.

    Returns
    -------
    pandas.DataFrame
      With columns "time" and a column for each outcome with the name of the outcome
    """
    merged = None
    for name, df in outcomes.items():
        if merged is None:
            # This returns a copy
            merged = df.rename(columns={"value" : name})
        else:
            merged = pd.merge(merged, df, on="time", how="left")
            # merge returns a copy, so we do not need to do a new copy
            # TODO: check if pandas copy-on-write implies that chaining merge and rename will only
            # result in one copy.
            merged.rename(columns={"value" : name}, inplace=True)
    return merged


def check_outcomes(outcomes):
    """Check that set of outcomes are valid and compatible. The point of the check is to ensure
    that every time point has a unique value and that we do not introduce NA values when merging.

    A DataFrame is valid if it has a "time" column with unique values and a "value" column

    Two outcomes A and B are compatible if
     - All time points in A are in B
     - All time points in B are in A

    Parameters
    ----------
    outcomes : { str : pandas.DataFrame }
      One or more outcomes to check
    """
    if len(outcomes) == 0:
        raise ValueError("No outcomes")

    time = None
    for df in outcomes.values():
        if time is None:
            # Full validate the first DataFrame
            _validate_outcome_dataframe(df)
            time = df["time"]
        else:
            # Skip uniqueness of timepoints because we check that they match
            _validate_outcome_dataframe(df, False)
            if len(df["time"]) != len(time) or not time.isin(df["time"]).all():
                raise ValueError("All DataFrames must have the same time points")

def _validate_outcome_dataframe(df, check_unique_timepoints=True):
    if "time" not in df.columns:
        raise ValueError("Missing 'time' column")
    if "value" not in df.columns:
        raise ValueError("Missing 'value' column")
    if len(df.columns) != 2:
        raise ValueError("Outcome DataFrames must have exactly two columns: 'time' and 'value'")
    if check_unique_timepoints:
        if df["time"].nunique() != len(df):
            raise ValueError("Time points are not unique")


def get_single_scalar(mapping):
    """Get the scalar value from a length 1 Mapping

    Parameters
    ----------
    mapping : Mapping
      A length 1 mapping with a scalar value

    Returns
    -------
    float

    Raises
    ------
    ValueError : If len(mapping) != 1 or the mapped value is not int or float
    """
    if len(mapping) != 1:
        raise ValueError("Expected a Mapping of length 1")
    value = list(mapping.values())[0]
    if not isinstance(value, (int, float)):
        raise ValueError("Expected a scalar value")
    return value


class StrictFormatter(Formatter):
    """Strict string formatting that fails if not all arguments are used for formatting

    Usage
    -----
      formatter = StrictFormatter()
      formatted = formatter.format(format_string, , /, *args, **kwargs)
    """
    def check_unused_args(self, used_args, args, kwargs):
        """Check if any arguments are unused

        Raises
        ------
        ValueError if there are unused arguments
        """
        potential_args = set(range(len(args))) | set(kwargs.keys())
        unused_args = potential_args - used_args
        if len(unused_args) > 0:
            raise ValueError(f"There are unusued format arguments: {unused_args}")
