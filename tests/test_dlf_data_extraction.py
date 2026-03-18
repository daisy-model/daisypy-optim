# pylint: disable=missing-function-docstring,R0801
from pathlib import Path
import pandas as pd
import pytest
from daisypy.optim import DlfDataExtractor, DlfSum
from daisypy.optim.dlf_data_extraction import DlfSingleton

EXPECTED_NO3 = pd.Series([
    0.000114649, 8.02047e-05, 5.61092e-05, 3.92532e-05, 2.74616e-05, 1.92128e-05, 1.34425e-05,
    9.40583e-06, 6.58209e-06, 4.60684e-06, 3.22518e-06, 2.25878e-06, 1.58289e-06, 1.11026e-06,
    7.79861e-07, 5.48966e-07, 3.87692e-07, 2.75131e-07, 1.96658e-07, 1.42039e-07, 1.04112e-07,
    7.78724e-08, 5.98114e-08, 4.74821e-08
])

EXPECTED_CO2 = pd.Series([
    -76.3192, -76.3185, -76.3178, -76.3171, -76.3164, -76.3157, -76.315	, -76.3143, -76.3136,
    -76.3129, -76.3122, -76.3115, -76.3108, -76.3101, -76.3094, -76.3087, -76.3079, -76.3072,
    -76.3065, -76.3058, -76.3051, -76.3044, -76.3037, -76.303
])

def test_single_var_no_post_processor():
    data_dir = Path(__file__).parent / 'test-data' / 'dlfs'
    extractor = DlfDataExtractor({'soil_NO3_profile.dlf' : 'NO3'})
    extracted = extractor(data_dir)
    assert set(extracted.columns) == { 'time' , 'value' }
    assert (EXPECTED_NO3 == extracted['value']).all()

def test_single_var_singleton_post_processor():
    data_dir = Path(__file__).parent / 'test-data' / 'dlfs'
    extractor = DlfDataExtractor({'soil_NO3_profile.dlf' : 'NO3'},  DlfSingleton())
    extracted = extractor(data_dir)
    assert set(extracted.columns) == { 'time' , 'value' }
    assert (EXPECTED_NO3 == extracted['value']).all()

def test_multiple_vars_no_post_processor():
    with pytest.raises(ValueError) as excinfo:
        _ = DlfDataExtractor({'soil_NO3_profile.dlf' : ['NO3', 'CO2']})
    assert "`logs_and_variables` must contain exactly one key" in str(excinfo.value)

def test_bad_post_processor():
    data_dir = Path(__file__).parent / 'test-data' / 'dlfs'
    extractor = DlfDataExtractor({'soil_NO3_profile.dlf' : 'NO3'},  lambda dfs: dfs[0])
    # Post processor does not ensure correct naming of columns, should raise an exception
    with pytest.raises(RuntimeError) as excinfo:
        _ = extractor(data_dir)
    assert "must return a DataFrame with columns 'time' and 'value'" in str(excinfo.value)

def test_multiple_vars_sum_post_processor():
    data_dir = Path(__file__).parent / 'test-data' / 'dlfs'
    extractor = DlfDataExtractor({'soil_NO3_profile.dlf' : ['NO3', 'CO2']}, DlfSum())
    extracted = extractor(data_dir)
    assert set(extracted.columns) == { 'time' , 'value' }
    assert (EXPECTED_NO3 + EXPECTED_CO2 == extracted['value']).all()

def test_bad_logs_and_variables():
    with pytest.raises(ValueError) as excinfo:
        # Pass what should be a dict as two separate params
        _ = DlfDataExtractor('soil_NO3_profile.dlf', 'NO3')
    assert '`logs_and_variables` must be a Mapping' in str(excinfo.value)
