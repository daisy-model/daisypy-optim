# pylint: disable=missing-function-docstring
import re
import pandas as pd
import pytest
from daisypy.optim.output_store import OutputStore

def test_output_store_insert_and_extract():
    store = OutputStore()
    time = pd.to_datetime(['2000-01-01', '2000-01-02'])

    store.insert('sim1', 'field_water.dlf', pd.DataFrame({
        'time' : time,
        'water' : [1.0, 3.0]
    }))
    store.insert('sim1', 'field_water.dlf', pd.DataFrame({
        'time' : time,
        'nitrogen' : [2.0, 4.0]
    }))
    store.insert('sim1', 'field_temp.dlf', pd.DataFrame({
        'time' : time,
        'temp' : [5.0, 7.0]
    }))

    water = store.extract('sim1', 'field_water.dlf', 'water')
    nitrogen = store.extract('sim1', 'field_water.dlf', 'nitrogen')
    expected_water = pd.DataFrame({
        'time' : time,
        'value' : [1.0, 3.0],
    })
    expected_nitrogen = pd.DataFrame({
        'time' : time,
        'value' : [2.0, 4.0],
    })
    pd.testing.assert_frame_equal(water, expected_water)
    pd.testing.assert_frame_equal(nitrogen, expected_nitrogen)


def test_output_store_insert_with_missing_time():
    store = OutputStore()
    time = pd.to_datetime(['2000-01-01', '2000-01-02'])

    with pytest.raises(ValueError, match="`df` MUST have a 'time' column"):
        store.insert('sim1', 'field_water.dlf', pd.DataFrame({
            'date' : time,
            'water' : [1.0, 3.0]
        }))

def test_output_store_insert_with_non_unique_timepoints():
    store = OutputStore()
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-01']),
        'water' : [1.0, 3.0]
    })
    with pytest.raises(ValueError, match="`df` MUST have unique timepoints"):
        store.insert('sim1', 'field_water.dlf', df)


def test_output_store_insert_with_timepoints_mismatch():
    store = OutputStore()
    df1 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'water' : [1.0, 3.0]
    })

    df2 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-03']),
        'water2' : [1.0, 3.0]
    })

    store.insert('sim1', 'field_water.dlf', df1)
    with pytest.raises(
            ValueError,
            match="Timepoints in `df` do not match existing timepoints in 'sim1/field_water.dlf'"):
        store.insert('sim1', 'field_water.dlf', df2)

def test_output_store_insert_with_same_value_name():
    store = OutputStore()
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'water' : [1.0, 3.0]
    })

    store.insert('sim1', 'field_water.dlf', df)
    msg = "Column(s) {'water'} already exists"
    with pytest.raises(ValueError, match=re.escape(msg)):
        store.insert('sim1', 'field_water.dlf', df)
