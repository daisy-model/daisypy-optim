# pylint: disable=missing-function-docstring
import pandas as pd
import pytest

from daisypy.optim.post_processors import AggregateOutcomes


def test_aggregate_outcomes_uses_declared_outcomes_and_returns_standard_outcome():
    time = pd.to_datetime(['2000-01-01', '2000-01-02'])
    outcomes = {
        'a' : pd.DataFrame({
            'time' : time,
            'value' : [1.0, 2.0],
        }),
        'b' : pd.DataFrame({
            'time' : time,
            'value' : [10.0, 20.0],
        }),
        'ignored' : pd.DataFrame({
            'time' : time,
            'value' : [100.0, 200.0],
        }),
    }

    def combine(values):
        assert list(values.index) == ['a', 'b']
        return values['a'] + values['b']

    aggregate = AggregateOutcomes(['a', 'b'], combine)

    result = aggregate(outcomes)

    expected = pd.DataFrame({
        'time' : time,
        'value' : [11.0, 22.0],
    })
    pd.testing.assert_frame_equal(result, expected)


def test_aggregate_outcomes_throws_when_names_mismatch():
    outcomes = {
        'a' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
            'value' : [1.0, 2.0],
        }),
    }
    aggregate = AggregateOutcomes(['a', 'b'], None)
    with pytest.raises(ValueError, match="`outcomes` does not contain 'b'"):
        aggregate(outcomes)
