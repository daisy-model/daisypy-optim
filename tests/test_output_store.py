# pylint: disable=missing-function-docstring
import numpy as np
import pandas as pd
from daisypy.optim.output_store import AggregateColumns, OutputStore


def test_output_store_from_empty_supports_insert_extract_and_combine():
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

    extracted = store.extract('sim1', 'field_water.dlf', ['water', 'nitrogen'])
    expected_extracted = pd.DataFrame({
        'time' : time,
        'water' : [1.0, 3.0],
        'nitrogen' : [2.0, 4.0]
    })
    pd.testing.assert_frame_equal(extracted, expected_extracted)

    store.combine(
        {
            'water' : ('sim1', 'field_water.dlf', 'water'),
            'temp' : ('sim1', 'field_temp.dlf', 'temp')
        },
        AggregateColumns(lambda values: np.sum(values, axis=0), 'combined'),
        ('sim1', 'combined.dlf')
    )

    expected_combined = pd.DataFrame({
        'time' : time,
        'combined' : [6.0, 10.0]
    })
    pd.testing.assert_frame_equal(store['sim1']['combined.dlf'], expected_combined)
