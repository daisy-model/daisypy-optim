# pylint: disable=missing-function-docstring
import pandas as pd
import pytest
from daisypy.optim.output_store import AggregateColumns, OutputStore, merge_outputs


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

    combined = store.combine(
        {
            'water' : ('sim1', 'field_water.dlf', 'water'),
            'temp' : ('sim1', 'field_temp.dlf', 'temp')
        },
        AggregateColumns(lambda row: row.sum(), 'combined')
    )

    expected_combined = pd.DataFrame({
        'time' : time,
        'combined' : [6.0, 10.0]
    })
    pd.testing.assert_frame_equal(combined, expected_combined)
    assert 'combined.dlf' not in store['sim1']


def test_merge_outputs_merges_named_dataframes_on_time():
    time = pd.to_datetime(['2000-01-01', '2000-01-02'])
    inputs = {
        'water' : pd.DataFrame({
            'time' : time,
            'water' : [1.0, 3.0]
        }),
        'temp' : pd.DataFrame({
            'time' : time,
            'temp' : [5.0, 7.0]
        })
    }

    merged = merge_outputs(inputs)

    expected = pd.DataFrame({
        'time' : time,
        'water' : [1.0, 3.0],
        'temp' : [5.0, 7.0]
    })
    pd.testing.assert_frame_equal(merged, expected)
