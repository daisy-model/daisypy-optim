import pandas as pd
from daisypy.optim.data_extraction import extract_from_dlf

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

    def combine(self, input_specs, combinator, output_spec):
        """Combine existing inputs and insert in to store

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
        self.insert(*output_spec, combinator(inputs))


class AggregateColumns:
    # pylint: disable=too-few-public-methods
    """Aggregate all non-time columns to produce a new DataFrame with aggregated values for each
    timepoint. All inputs MUST have the same timepoints.
    """
    def __init__(self, fn, out_name):
        """
        Parameters
        ----------
        fn : callable [[numpy.ndarray]] -> numpy.ndarray
          Callable aggregating the column values into a single column of values

        out_name : str
          Name to use for the output column
        """
        self.fn = fn
        self.out_name = out_name

    def __call__(self, inputs):
        """Aggregate the inputs

        Parameters
        ----------
        inputs : { str : pandas.DataFrame }
          dict of input names to input DataFrames. The names are ingored

        Returns
        -------
        pandas.DataFrame with columns "time" and out_name
        """
        merged = None
        for df in inputs.values():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="time", validate="1:1")
        # Merging worked
        values = [merged[c].values for c in merged.columns if c != "time"]
        return pd.DataFrame({
            "time" : merged["time"],
            self.out_name : self.fn(values)
        })
