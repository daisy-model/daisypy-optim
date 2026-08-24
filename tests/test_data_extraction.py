# pylint: disable=missing-function-docstring,R0801
from pathlib import Path
import pandas as pd
from daisypy.optim.data_extraction import extract_from_dlf
from daisypy.optim.output_spec import OutputSpec

EXPECTED_NO3 = pd.Series([
    0.000114649, 8.02047e-05, 5.61092e-05, 3.92532e-05, 2.74616e-05, 1.92128e-05, 1.34425e-05,
    9.40583e-06, 6.58209e-06, 4.60684e-06, 3.22518e-06, 2.25878e-06, 1.58289e-06, 1.11026e-06,
    7.79861e-07, 5.48966e-07, 3.87692e-07, 2.75131e-07, 1.96658e-07, 1.42039e-07, 1.04112e-07,
    7.78724e-08, 5.98114e-08, 4.74821e-08
])

EXPECTED_CO2 = pd.Series([
    -76.3192, -76.3185, -76.3178, -76.3171, -76.3164, -76.3157, -76.315, -76.3143, -76.3136,
    -76.3129, -76.3122, -76.3115, -76.3108, -76.3101, -76.3094, -76.3087, -76.3079, -76.3072,
    -76.3065, -76.3058, -76.3051, -76.3044, -76.3037, -76.303
])


def _test_data_dir():
    return Path(__file__).parent / 'test-data' / 'dlfs'


def test_extract_from_dlf_extracts_single_outputspec_variable():
    data_dir = _test_data_dir()
    extracted = extract_from_dlf({
        'field' : OutputSpec('soil_NO3_profile.dlf', 'NO3', root=data_dir)
    })

    assert set(extracted.keys()) == {'field'}
    assert list(extracted['field'].columns) == ['time', 'NO3']
    assert (EXPECTED_NO3 == extracted['field']['NO3']).all()


def test_extract_from_dlf_extracts_multiple_variables_from_single_file():
    data_dir = _test_data_dir()
    extracted = extract_from_dlf({
        'field' : OutputSpec('soil_NO3_profile.dlf', ['NO3', 'CO2'], root=data_dir)
    })

    assert set(extracted.keys()) == {'field'}
    assert list(extracted['field'].columns) == ['time', 'NO3', 'CO2']
    assert (EXPECTED_NO3 == extracted['field']['NO3']).all()
    assert (EXPECTED_CO2 == extracted['field']['CO2']).all()


def test_extract_from_dlf_extracts_multiple_named_outputs():
    data_dir = _test_data_dir()
    extracted = extract_from_dlf({
        'no3' : OutputSpec('soil_NO3_profile.dlf', 'NO3', root=data_dir),
        'co2' : OutputSpec('soil_NO3_profile.dlf', 'CO2', root=data_dir)
    })

    assert set(extracted.keys()) == {'no3', 'co2'}
    assert list(extracted['no3'].columns) == ['time', 'NO3']
    assert list(extracted['co2'].columns) == ['time', 'CO2']
    pd.testing.assert_series_equal(extracted['no3']['time'], extracted['co2']['time'])
    assert (EXPECTED_NO3 == extracted['no3']['NO3']).all()
    assert (EXPECTED_CO2 == extracted['co2']['CO2']).all()
