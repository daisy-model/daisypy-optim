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

    def extract(self, spec):
        """Get a specific output variable

        Parameters
        ----------
        spec : daisypy.optim.output_spec.OutputSpec
          Specification of the output to get.

        Returns
        -------
        pandas.DataFrame with columns "time" and "value"
        """
        return self[spec.sim][spec.output][["time", spec.var]].rename(columns={spec.var:"value"})

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
