# pylint: disable=missing-function-docstring
import pandas as pd
import pytest
from daisypy.optim.util import StrictFormatter, check_target, check_outcomes, merge_outcomes


def test_check_target_valid():
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    check_target(df, 'a')

def test_check_target_missing_time():
    df = pd.DataFrame({
        'date' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    with pytest.raises(ValueError, match="Missing 'time' column"):
        check_target(df, 'a')

def test_check_target_missing_target_col():
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    with pytest.raises(ValueError, match="Missing 'b' column"):
        check_target(df, 'b')

def test_check_target_non_unique_timepoints():
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-01']),
        'a' : [1.0, 2.0]
    })
    with pytest.raises(ValueError, match="Time points are not unique"):
        check_target(df, 'a')


def test_check_outcomes_single_valid():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
        })
    }
    check_outcomes(outcomes)

def test_check_outcomes_multiple_valid():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
        }),
        'b' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [3.0, 4.0],
        }),
        'c' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [5.0, 6.0],
        })
    }
    check_outcomes(outcomes)

def test_check_outcomes_missing_time():
    outcomes = {
        'a' : pd.DataFrame({
            'date' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
        })
    }
    with pytest.raises(ValueError, match="Missing 'time' column"):
        check_outcomes(outcomes)

def test_check_outcomes_missing_value():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'outcome' : [1.0, 2.0],
        })
    }
    with pytest.raises(ValueError, match="Missing 'value' column"):
        check_outcomes(outcomes)

def test_check_outcomes_extra_column():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
            'meta' : [3.0, 4.0],
        })
    }
    with pytest.raises(
            ValueError,
            match="Outcome DataFrames must have exactly two columns: 'time' and 'value'"):
        check_outcomes(outcomes)


def test_check_outcomes_mismatched_time_points():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0]
        }),
        'b' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-03']),
            'value' : [3.0, 4.0]
        })
    }

    with pytest.raises(ValueError, match='All DataFrames must have the same time points'):
        check_outcomes(outcomes)

def test_merge_outcomes_single_valid():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
        }),
    }
    df = merge_outcomes(outcomes)
    expected = pd.DataFrame({
        "time" : pd.to_datetime(['2000-01-01', '2000-01-02']),
        "a" : [1.0, 2.0],
    })
    pd.testing.assert_frame_equal(df, expected)

def test_merge_outcomes_multiple_valid():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
        }),
        'b' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [3.0, 4.0],
        }),
        'c' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [5.0, 6.0],
        })
    }
    df = merge_outcomes(outcomes)
    expected = pd.DataFrame({
        "time" : pd.to_datetime(['2000-01-01', '2000-01-02']),
        "a" : [1.0, 2.0],
        "b" : [3.0, 4.0],
        "c" : [5.0, 6.0]
    })
    pd.testing.assert_frame_equal(df, expected)


def test_strict_formatter_rejects_unused_arguments():
    formatter = StrictFormatter()

    assert formatter.format('{a}-{b}', a='left', b='right') == 'left-right'

    with pytest.raises(ValueError, match='unused format arguments'):
        formatter.format('{a}', a='left', b='right')
