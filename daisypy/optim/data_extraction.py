# pylint: disable=too-few-public-methods
import pandas as pd
from daisypy.io.dlf import read_dlf

def extract_from_dlf(output_specs):
    """Extract data from dlf files defined in OutputSpecs

    Parameters
    ----------
    output_specs : dict of (str, OutputSpec)
      Dict mapping output name to output specification

    Returns
    -------
    dict of (str, pandas.DataFrame)
      Dict mapping output name to extracted data
    """
    extracted = {}
    for name, output_spec in output_specs.items():
        dlf = read_dlf(output_spec.path())
        dlf.body["time"] = pd.to_datetime(
            dlf.body[["year", "month", "mday", "hour"]].rename(columns={"mday" : "day"})
        )
        extracted[name] = dlf.body[["time"] + output_spec.var]
    return extracted
