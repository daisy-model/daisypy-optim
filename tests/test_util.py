# pylint: disable=missing-function-docstring
import pandas as pd
import pytest
from daisypy.optim.util import check_dataframes


def test_check_dataframes_accepts_valid_single_and_multiple_frames():
    df1 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    df2 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-02', '2000-01-01']),
        'b' : [3.0, 4.0]
    })

    check_dataframes(df1)
    check_dataframes(df1, df2)

def test_check_dataframes_accepts_multiple_columns():
    df1 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0],
        'c' : [0.1, 0.2]
    })
    df2 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-02', '2000-01-01']),
        'b' : [3.0, 4.0],
        'd' : [0.3, 0.4]
    })

    check_dataframes(df1)
    check_dataframes(df1, df2)

    

def test_check_dataframes_rejects_missing_time_column():
    df = pd.DataFrame({'a' : [1.0, 2.0]})

    with pytest.raises(ValueError, match="Missing 'time' column"):
        check_dataframes(df)


def test_check_dataframes_rejects_missing_value_columns():
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02'])
    })

    with pytest.raises(ValueError, match='No value column\\(s\\)'):
        check_dataframes(df)

    with pytest.raises(ValueError, match='No value column\\(s\\)'):
        check_dataframes(
            pd.DataFrame({
                'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
                'a' : [1.0, 2.0]
            }),
            df
        )


def test_check_dataframes_rejects_duplicate_time_points():
    df = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-01']),
        'a' : [1.0, 2.0]
    })

    with pytest.raises(ValueError, match='Time points are not unique'):
        check_dataframes(df)


def test_check_dataframes_rejects_mismatched_time_points():
    df1 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    df2 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-03']),
        'b' : [3.0, 4.0]
    })

    with pytest.raises(ValueError, match='All DataFrames must have the same time points'):
        check_dataframes(df1, df2)


def test_check_dataframes_rejects_duplicate_non_time_columns_by_default():
    df1 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    df2 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [3.0, 4.0]
    })

    with pytest.raises(ValueError, match='Non time columns must be unique across DataFrames'):
        check_dataframes(df1, df2)


def test_check_dataframes_allows_duplicate_non_time_columns_when_disabled():
    df1 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-01', '2000-01-02']),
        'a' : [1.0, 2.0]
    })
    df2 = pd.DataFrame({
        'time' : pd.to_datetime(['2000-01-02', '2000-01-01']),
        'a' : [3.0, 4.0]
    })

    check_dataframes(df1, df2, check_unique_col_names=False)
