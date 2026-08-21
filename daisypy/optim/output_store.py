import pandas as pd
from daisypy.optim.data_extraction import extract_from_dlf
from daisypy.optim.util import merge_dataframes, check_dataframes

class OutputStore(dict):
    """An output store is an extended dict that holds outputs from a Daisy simulation

    Outputs are stored as pandas.DataFrame, one DataFrame for each simulation/output combination.
    """

    def __init__(self, simulations=None):
        """
        Parameters
        ----------
        simulations : { str : Simulation } or None
          Dict of named simulations
        """
        if simulations is None:
            super().__init__({})
        else:
            super().__init__({
                k : extract_from_dlf(sim.outputs) for k, sim in simulations.items()
            })

    def extract(self, sim, output, var):
        """Get a specific output variable

        Parameters
        ----------
        sim : str
          The simulation to get data from

        output : str
          The output to get data from

        var : str or [str]
          The specific column(s) to get

        Returns
        -------
        pandas.DataFrame with columns "time" and var
        """
        if isinstance(var, list):
            return self[sim][output][["time"] + var]
        return self[sim][output][["time", var]]

    def insert(self, sim, output, df):
        """Insert new values in the store

        Parameters
        ----------
        sim : str
          Name of simulation to store value under. May be new or existing

        output : str
          Name of output to store value under. May be new or existing.

        df : pandas.DataFrame
          Values to store. Must have column "time".
          If sim/output already contains a DataFrame then the timepoints must match and no column
          in df may be in the existing DataFrame.
        """
        assert "time" in df, "df MUST have a 'time' column"
        if not sim in self:
            self[sim] = { output : df }
        elif not output in self[sim]:
            self[sim][output] = df
        else:
            for c in df.columns:
                if c != "time":
                    assert c not in self[sim][output].columns, f"{c} is already in {sim}/{output}"
            self[sim][output] = pd.merge(self[sim][output], df, on="time", validate="1:1")

    def combine(self, input_specs, combinator):
        """Combine inputs in the store and return the result

        Parameters
        ----------
        input_specs : [(str, str, str OR [str])]
          List of inputs to combine, each input spec is a triple of (sim, output, var) and must
          exist in the store. Note that var can be a list of strings if several columns from the
          output file is needed.

        combinator : callable [{str : pandas.DataFrame}] -> pandas.DataFrame
          Callable combining the inputs. Input DataFrames have "time" column and the requested
          variables. Output DataFrame must have "time" column and whatever columns are computed.
        """
        inputs = {}
        for k, spec in input_specs.items():
            inputs[k] = self.extract(*spec)
        return combinator(inputs)


def merge_outputs(inputs):
    """Merge output DataFrames. All inputs MUST have "time" column with the same timepoints and
    no inputs may share other column names

    Parameters
    ----------
    inputs : { str : pandas.DataFrame }
      Named DataFrames. The names are ignored

    Returns
    -------
    pandas.DataFrame with all inputs merged
    """
    inputs = list(inputs.values())
    check_dataframes(*inputs) # Will throw if there are issues
    return merge_dataframes(*inputs)


class AggregateColumns:
    # pylint: disable=too-few-public-methods
    """Aggregate all non-time columns to produce a new DataFrame with aggregated values for each
    timepoint. All inputs MUST have the same timepoints.
    """
    def __init__(self, fn):
        """
        Parameters
        ----------
        fn : Callable [pandas.Series] -> float
          Scalar valued aggregation function.

        out_name : str
          Name to use for the output column
        """
        self.fn = fn

    def __call__(self, inputs):
        """Aggregate the inputs

        Parameters
        ----------
        inputs : { str : pandas.DataFrame }
          Named DataFrames. The names are ignored

        Returns
        -------
        pandas.DataFrame with columns "time" and out_name
        """
        inputs = list(inputs.values())
        check_dataframes(*inputs) # Will throw if there are issues
        merged = merge_dataframes(*inputs)
        return pd.DataFrame({
            "time" : merged["time"],
            "value" : merged.drop(columns=["time"]).aggregate(self.fn, axis="columns").values
        })
