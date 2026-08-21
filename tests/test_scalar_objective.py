# pylint: disable=missing-function-docstring
from pathlib import Path
import pandas as pd
import pytest
from daisypy.optim.scalar_objective import ScalarObjective


def mse(actual, target):
    return ((actual - target) ** 2).mean()


def _expected_target():
    in_dir = Path(__file__).parent / 'test-data' / 'targets'
    expected = pd.read_csv(in_dir / 'comma-separated.csv').rename(columns={'NO3' : 'value'})
    expected['time'] = pd.to_datetime(expected['time'])
    return expected


def test_scalar_objective_accepts_csv_targets_with_auto_delimiter_detection():
    in_dir = Path(__file__).parent / 'test-data' / 'targets'
    expected = _expected_target()
    outcomes = {'prediction' : expected}

    for target_file in in_dir.iterdir():
        if target_file.name.endswith('separated.csv'):
            objective = ScalarObjective(
                target_file.name,
                target_file,
                'NO3',
                'prediction',
                mse
            )
            assert objective(outcomes) == {target_file.name : 0}


def test_scalar_objective_accepts_dataframe_target_and_returns_named_loss():
    target = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'target_value' : [1.0, 3.0]
    })
    outcomes = {
        'prediction' : pd.DataFrame({
            'time' : pd.to_datetime(['2000-01-03', '2000-01-01', '2000-01-02']),
            'value' : [999.0, 1.0, 3.0]
        })
    }

    objective = ScalarObjective('objective', target, 'target_value', 'prediction', mse)

    assert objective(outcomes) == {'objective' : 0}


def test_scalar_objective_rejects_missing_target_column():
    target = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'other' : [1.0, 3.0]
    })

    with pytest.raises(ValueError, match='target must contain "missing" column'):
        ScalarObjective('objective', target, 'missing', 'prediction',  mse)
