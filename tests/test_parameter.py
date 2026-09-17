# pylint: disable=missing-function-docstring
import re
import pytest
from daisypy.optim.parameter import CategoricalParameter, ContinuousParameter

def test_continuous_parameter_sample_raises():
    p1 = ContinuousParameter('p1', 0.5, (0, 1))
    with pytest.raises(ValueError, match=re.escape('`num_samples` must be positive (0)')):
        p1.sample(0)

def test_continuous_parameter_sample_one():
    p1 = ContinuousParameter('p1', 0.5, (0, 1))
    assert p1.sample(1) == [0.5]

def test_continuous_parameter_sample_two():
    p1 = ContinuousParameter('p1', 0.4, (0, 1))
    assert p1.sample(2) == [0.4, 1]

    p2 = ContinuousParameter('p2', 0.6, (0, 1))
    assert p2.sample(2) == [0.6, 0]

def test_continuous_parameter_sample_three():
    p1 = ContinuousParameter('p1', 0.01, (0, 1))
    assert p1.sample(3) == [0.01, 0, 1]

def test_continuous_parameter_sample_init_at_center():
    p1 = ContinuousParameter('p1', 0.5, (0, 1))
    samples = p1.sample(11)
    expected = [0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
    assert pytest.approx(samples) == expected

def test_continuous_parameter_sample_init_off_center():
    p1 = ContinuousParameter('p1', 0.8, (0, 1))
    samples = p1.sample(11)
    expected = [0.8, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]
    assert pytest.approx(samples) == expected

def test_continuous_parameter_as_categorical():
    p1 = ContinuousParameter('p1', 0.8, (0, 1))
    p1_cat = p1.as_categorical(11)
    assert isinstance(p1_cat, CategoricalParameter)
    assert p1_cat.name == p1.name
    assert p1_cat.initial_value_idx == 0
    expected = [0.8, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]
    assert pytest.approx(p1_cat.values) == expected
