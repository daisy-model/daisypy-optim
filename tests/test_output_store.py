# pylint: disable=missing-function-docstring
import pandas as pd
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
