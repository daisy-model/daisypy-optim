# pylint: disable=missing-function-docstring
from pathlib import Path
import pandas as pd
from daisypy.optim import ScalarObjective
from daisypy.optim.loss_fns import mse
from .mockup import MockDataExtractor

def test_csv_delimiter():
    in_dir = Path(__file__).parent / 'test-data' / 'targets'
    expected = pd.read_csv(in_dir / 'comma-separated.csv').rename(columns={"NO3" : "value"})
    expected["time"] = pd.to_datetime(expected["time"])
    data_extractor = MockDataExtractor(expected)
    for target_file in in_dir.iterdir():
        if target_file.name.endswith('separated.csv'):
            # We use the same extracted data, but change the target each time.
            f = ScalarObjective(target_file.name, data_extractor, target_file, "NO3", mse)
            result = f(in_dir).pop(target_file.name)
            assert result == 0, target_file


def test_scalar_objective_evaluate_returns_exact_prediction():
    in_dir = Path(__file__).parent / 'test-data' / 'targets'
    expected = pd.read_csv(in_dir / 'comma-separated.csv').rename(columns={"NO3" : "value"})
    expected["time"] = pd.to_datetime(expected["time"])
    data_extractor = MockDataExtractor(expected)

    objective = ScalarObjective('NO3', data_extractor, in_dir / 'comma-separated.csv', 'NO3', mse)
    evaluation = objective.evaluate(in_dir)

    assert evaluation.objectives['NO3'] == 0
    pd.testing.assert_frame_equal(evaluation.predictions['NO3'], expected)
