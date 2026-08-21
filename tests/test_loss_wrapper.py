# pylint: disable=missing-function-docstring
import pandas as pd
import pytest
from daisypy.optim.loss_wrapper import LossWrapper


def mse(actual, target):
    return ((actual - target) ** 2).mean()


def test_loss_wrapper_drops_missing_targets_and_ignores_extra_actual_timestamps():
    actual = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02', '2000-01-03']),
        'value' : [1.0, 2.0, 999.0]
    })
    target = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'value' : [1.0, None]
    })

    loss = LossWrapper(mse)(actual=actual, target=target)

    assert loss == 0


def test_loss_wrapper_rejects_duplicate_target_timestamps():
    actual = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'value' : [1.0, 2.0]
    })
    target = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-01']),
        'value' : [1.0, 1.0]
    })

    with pytest.raises(ValueError, match='Timestamps in target must be unique'):
        LossWrapper(mse)(actual=actual, target=target)


def test_loss_wrapper_rejects_duplicate_actual_timestamps():
    actual = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-01']),
        'value' : [1.0, 1.0]
    })
    target = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01']),
        'value' : [1.0]
    })

    with pytest.raises(ValueError, match='Timestamps in actual must be unique'):
        LossWrapper(mse)(actual=actual, target=target)


def test_loss_wrapper_rejects_missing_actual_timestamps():
    actual = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01']),
        'value' : [1.0]
    })
    target = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'value' : [1.0, 2.0]
    })

    with pytest.raises(ValueError, match='All timestamps in target must be in actual'):
        LossWrapper(mse)(actual=actual, target=target)
